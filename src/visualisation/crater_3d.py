"""
3D crater visualisation — LOLA DEM + ice probability overlay + traverse corridors.
Requires: numpy, matplotlib, rasterio, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom
import rasterio

def load_dem_patch(dem_path: str, row_off: int, col_off: int,
                   height: int = 512, width: int = 512) -> np.ndarray:
    with rasterio.open(dem_path) as src:
        window = rasterio.windows.Window(col_off, row_off, width, height)
        return src.read(1, window=window).astype(float)

def render_crater_3d(dem: np.ndarray,
                     ice_prob: np.ndarray,
                     traverse_paths: list,
                     pixel_size_m: float = 15.0,
                     save_path: str = "results/crater_3d.png",
                     downsample: int = 2):
    """
    Render LOLA DEM with ice probability overlay and traverse corridors.

    Parameters
    ----------
    dem           : 2D elevation array (metres)
    ice_prob      : 2D ice probability [0,1], same shape as dem
    traverse_paths: list of (N,2) arrays of (row,col) path coordinates
    pixel_size_m  : ground resolution in metres
    save_path     : output PNG path
    downsample    : factor to downsample for rendering speed
    """
    # Downsample for rendering
    d = downsample
    dem_d  = zoom(dem, 1/d, order=1)
    ip_d   = zoom(ice_prob, 1/d, order=1)

    H, W = dem_d.shape
    X = np.arange(W) * pixel_size_m * d
    Y = np.arange(H) * pixel_size_m * d
    X, Y = np.meshgrid(X, Y)

    # Ice probability → RGBA overlay (blue = ice-likely)
    colormap = cm.RdYlBu_r
    face_colours = colormap(ip_d)

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, dem_d,
                    facecolors=face_colours,
                    linewidth=0, antialiased=True,
                    alpha=0.85)

    # Draw traverse corridors
    colours = ['lime', 'yellow', 'cyan']
    for idx, path in enumerate(traverse_paths[:3]):
        px = path[:, 1] / d * pixel_size_m * d
        py = path[:, 0] / d * pixel_size_m * d
        # Interpolate elevation along path
        pz = dem_d[np.clip((path[:,0]//d).astype(int), 0, H-1),
                   np.clip((path[:,1]//d).astype(int), 0, W-1)] + 20
        ax.plot(px, py, pz,
                color=colours[idx % len(colours)],
                linewidth=2.5,
                label=f'Corridor {idx+1}')

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('Lunar DSC: Ice Probability Overlay + Rover Traverse Corridors')
    ax.legend(loc='upper right')

    # Colourbar (ice probability)
    sm = plt.cm.ScalarMappable(cmap=colormap,
                                norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.1,
                 label='Ice Probability $\\Pi(p)$')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    # Smoke test with synthetic data
    rng = np.random.default_rng(42)
    dem_fake = np.zeros((256, 256))
    # Simulate crater bowl
    cx, cy = 128, 128
    for i in range(256):
        for j in range(256):
            r = np.sqrt((i-cx)**2 + (j-cy)**2)
            dem_fake[i,j] = -max(0, 50 - r) * 3 + rng.normal(0, 2)

    ip_fake = np.zeros((256,256))
    ip_fake[100:160, 100:160] = rng.uniform(0.6, 0.9, (60,60))  # ice patch

    path_fake = [np.column_stack([
        np.linspace(10, 128, 50),
        np.linspace(10, 128, 50)
    ]).astype(int)]

    import os; os.makedirs("results", exist_ok=True)
    render_crater_3d(dem_fake, ip_fake, path_fake,
                     save_path="results/crater_3d_test.png")
