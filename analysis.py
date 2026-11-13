import numpy as np
from constants import *
import matplotlib.pyplot as plt

def total_energy(system):
    Ek = 0.0
    Ep = 0.0
    for i in range(len(system.bodies)):
        velocity = np.linalg.norm(system.velocities[i])
        Ek += system.masses[i]*velocity**2/2
    for i in range(len(system.bodies)):
        for j in range(i + 1, len(system.bodies)):
            r_ij = np.linalg.norm(system.positions[i] - system.positions[j])
            Ep += -G*system.masses[i]*system.masses[j]/r_ij
    E = Ep + Ek
    return E

def run(system, method, dt, n_steps):
    energies = [total_energy(system)]
    times = [0]
    current = system
    for i in range(n_steps):
        times.append((i + 1)*dt)
        current = method.step(current, dt)
        energies.append(total_energy(current))
    return times, energies

def plot_total_energy(times, energies, name):
    x = times
    y = [energy/energies[0] for energy in energies]
    c = ''
    if name == "Euler": c = 'r'
    elif name == "RK4": c = 'b'
    elif name == "Verlet": c = 'g'
    plt.plot(x, y, c, label = name)
    plt.title(f"Graph of deviation of energy over time")
    plt.xlabel("t")
    plt.ylabel("E[t]/E[0]")
    plt.legend()
    plt.grid(True)



