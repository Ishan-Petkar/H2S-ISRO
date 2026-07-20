import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tifffile
import os
import sys

# Load pipeline functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline_skeleton import astar_traverse
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

path_coords = None
if output.feasible:
    path_coords = output.paths[0][0]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

H, W = surf.shape
x = np.arange(W) * pixel_size_m
y = np.arange(H) * pixel_size_m
X, Y = np.meshgrid(x, y)

# Normalize cost map for coloring the surface
norm_cost = (cost_map - cost_map.min()) / (cost_map.max() - cost_map.min())

surf_plot = ax.plot_surface(X, Y, surf, facecolors=plt.cm.RdYlGn_r(norm_cost),
                       linewidth=0, antialiased=False, alpha=0.8)

if path_coords is not None:
    path_coords = np.array(path_coords)
    px = path_coords[:, 1] * pixel_size_m
    py = path_coords[:, 0] * pixel_size_m
    pz = surf[path_coords[:, 0], path_coords[:, 1]] + 5.0 # Hover slightly above ground
    ax.plot(px, py, pz, color='cyan', linewidth=4, label='A* Optimal Path')
    
    # Plot start and end
    ax.scatter([px[0]], [py[0]], [pz[0]], color='white', s=100, label='Start (Lander)')
    ax.scatter([px[-1]], [py[-1]], [pz[-1]], color='magenta', s=100, label='Target (Ice)')

ax.set_title('3D Rover Traverse on LOLA DEM (Colored by Risk/Slope Cost)', fontsize=14)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Elevation (m)')

# Add a colorbar for the cost
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=cost_map.min(), vmax=cost_map.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
cbar.set_label('A* Traversal Cost $C(p)$')

if path_coords is not None:
    ax.legend()

ax.view_init(elev=45, azim=135)
plt.savefig('results/traverse_3d.pdf', format='pdf', bbox_inches='tight', dpi=300)
print("Successfully generated traverse_3d.pdf")
