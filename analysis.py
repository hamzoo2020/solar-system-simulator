import numpy as np
from constants import *

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

