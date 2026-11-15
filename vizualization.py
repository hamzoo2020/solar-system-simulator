import numpy as np
import matplotlib.pyplot as plt
from constants import *

def plot_orbits_2d(system, method, dt, n_steps, save_path="orbits.png"):
    positions_history = [system.positions]
    current = system
    for i in range(n_steps):
        current = method.step(current, dt)
        positions_history.append(current.positions.copy())

    plt.figure(figsize=(10, 10))
    colors = ['gray', 'orange', 'yellow', 'blue', 'red', 'brown', 'gold', 'cyan', 'navy']
    for i in range(len(system.bodies)):
        body = system.bodies[i]
        if i == 0:
            plt.plot(0, 0, 'o', color='yellow', markersize=5, label="Sun")
            continue
        xs = [pos[i][0] / AU for pos in positions_history]
        ys = [pos[i][1] / AU for pos in positions_history]
        plt.plot(xs, ys, color=colors[i], label=body.name, linewidth=1) #linewidth 1 ili 2

    plt.xlabel("x[AU]")
    plt.ylabel("y[AU]")
    plt.title("2D vizualization")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.show()

