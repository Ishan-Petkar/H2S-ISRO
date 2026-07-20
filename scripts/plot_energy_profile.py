import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
import sys

# Load pipeline functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline_skeleton import astar_traverse, compute_energy
from src.planner_execution import load_real_lola_dem

os.makedirs('results', exist_ok=True)

try:
    surf, slp = load_real_lola_dem((200, 200))
except Exception as e:
    print(f"Error loading real DEM: {e}")
    sys.exit(1)

# Mock cost map similar to planner_execution.py
pi_map = np.zeros_like(surf)
phi_map = np.zeros_like(surf)
rho_map = np.zeros_like(surf)

cost_map = (
    0.35 * (1.0 - pi_map) +
    0.40 * (slp / 30.0) +
    0.15 * phi_map +
    0.10 * (rho_map / 0.1)
)

start_pos = (5, 5)
targets = np.array([[190, 190]])
e_max_wh = 5000.0
t_relay_hours = 1.63
pixel_size_m = 5.0

output = astar_traverse(
    cost_map=cost_map,
    start=start_pos,
    targets=targets,
    e_max_wh=e_max_wh,
    t_relay_hours=t_relay_hours,
    pixel_size_m=pixel_size_m,
    slope_map=slp,
    contingency=0.20
)

if not output.feasible:
    print("Path not feasible!")
    sys.exit(1)

path_coords = np.array(output.paths[0][0])

cum_energy = [0.0]
for i in range(1, len(path_coords)):
    # Calculate distance for this slice
    sub_path = path_coords[:i+1]
    
    # Calculate energy up to step i
    # We pass n_science_stops=0, t_shadow_hours based on distance
    dist = i * pixel_size_m  # Approx distance
    t_shadow = dist / 0.1 / 3600.0
    
    e_dict = compute_energy(sub_path, slp, pixel_size_m, 0, t_shadow)
    # Add contingency just like planner
    total = sum(e_dict.values())
    cum_energy.append(total * 1.20)

cum_energy = np.array(cum_energy)
distances = np.arange(len(path_coords)) * pixel_size_m

plt.figure(figsize=(10, 6))
plt.plot(distances, cum_energy, color='red', linewidth=3, label='Cumulative Energy (w/ 20% margin)')
plt.axhline(5000.0, color='black', linestyle='--', linewidth=2, label='Absolute Max Battery ($E_{max}=5000$ Wh)')
plt.axhline(5000.0 * 0.8, color='orange', linestyle='--', linewidth=2, label='Operational Limit ($4000$ Wh)')

plt.fill_between(distances, cum_energy, 4000.0, where=(cum_energy < 4000.0), color='green', alpha=0.1, label='Safe Margin')

plt.xlabel('Distance Traveled (m)', fontsize=14)
plt.ylabel('Energy Consumed (Wh)', fontsize=14)
plt.title('Rover Energy Consumption Profile During A* Traverse', fontsize=16)

plt.xlim(0, max(distances))
plt.ylim(0, 5500)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig('results/energy_profile.pdf', format='pdf', bbox_inches='tight')
print("Successfully generated energy_profile.pdf")
