from constants import *
from math import sqrt
import numpy as np

class System:
    __slots__ = ("bodies", "positions", "velocities", "masses")

    def __init__(self, bodies, positions, velocities):
        self.bodies = bodies
        self.positions = positions
        self.velocities = velocities
        self.masses = np.array([body.mass for body in self.bodies])

    def accelerations(self, positions = None):
        if positions is None:
            positions = self.positions
        N = len(self.masses)
        a = np.zeros((N, 3))
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                r_ij = positions[j] - positions[i];
                d = np.linalg.norm(r_ij)
                a[i] += G*self.masses[j]*r_ij / d**3
        return a    