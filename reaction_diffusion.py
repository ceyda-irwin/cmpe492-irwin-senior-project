import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid size
N = 200

# Gray-Scott parameters
Du = 0.16   # diffusion rate of U
Dv = 0.08   # diffusion rate of V
F = 0.060   # feed rate
k = 0.062   # kill rate

# Time step
dt = 1.0

# Initialize concentration grids
U = np.ones((N, N), dtype=np.float64)
V = np.zeros((N, N), dtype=np.float64)

# Add a small square disturbance in the center
r = 20
center = N // 2
U[center-r:center+r, center-r:center+r] = 0.50
V[center-r:center+r, center-r:center+r] = 0.25

# Add some random noise
noise = 0.02
U += noise * np.random.random((N, N))
V += noise * np.random.random((N, N))

# Keep values in valid range
U = np.clip(U, 0, 1)
V = np.clip(V, 0, 1)


def laplacian(Z: np.ndarray) -> np.ndarray:
    """
    Compute Laplacian using 2D finite differences
    with periodic boundary conditions via np.roll.
    """
    return (
        -4 * Z
        + np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
    )


def update(_: int) -> list:
    global U, V

    # Run several simulation steps per frame for faster visible evolution
    for _ in range(10):
        Lu = laplacian(U)
        Lv = laplacian(V)

        uvv = U * V * V

        U += (Du * Lu - uvv + F * (1 - U)) * dt
        V += (Dv * Lv + uvv - (F + k) * V) * dt

        U = np.clip(U, 0, 1)
        V = np.clip(V, 0, 1)

    img.set_array(V)
    return [img]


# Visualization
fig, ax = plt.subplots(figsize=(6, 6))
img = ax.imshow(V, cmap="plasma", interpolation="nearest")
ax.set_title("Gray-Scott Reaction-Diffusion")
ax.axis("off")

ani = FuncAnimation(fig, update, interval=50, blit=True)
plt.show()