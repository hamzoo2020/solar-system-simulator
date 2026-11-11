from math import sqrt
from system import System
import numpy as np

class Integrator:
    def step(self, system, dt):
        raise NotImplementedError

class Euler:
    def step(self, system, dt):
        a = system.accelerations()
        new_positions = system.positions + system.velocities * dt
        new_velocities = system.velocities + a * dt
        return System(system.bodies, new_positions, new_velocities)

class RK4:
    def step(self, system, dt):
        k1_p = system.velocities
        k1_v = system.accelerations()
        k2_p = system.velocities + (dt/2)*k1_v
        k2_v = system.accelerations(system.positions + (dt/2)*k1_p)
        k3_p = system.velocities + (dt/2)*k2_v
        k3_v = system.accelerations(system.positions + (dt/2)*k2_p)
        k4_p = system.velocities + dt*k3_v
        k4_v = system.accelerations(system.positions + dt*k3_p)
        new_velocities = system.velocities + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
        new_positions = system.positions + (dt/6)*(k1_p + 2*k2_p + 2*k3_p + k4_p)
        return System(system.bodies, new_positions, new_velocities)
        '''for i in range(len(system.velocities)):
            v = system.velocities[i]
            k1_p = v
            k1_v = system.accelerations(system.positions[i])
            k2_p = v + dt/2*k1_v
            k2_v = system.accelerations(k2_p)*(system.positions[i] + (dt/2) * k1_p)
            k3_p = v + (dt/2) * k2_v
            k3_v = system.accelerations(k3_p)*(system.positions[i] + (dt/2) * k2_p)
            k4_p = v + dt * k3_v
            k4_v = system.accelerations(k4_p)*(system.positions[i] + dt * k3_p)
            new_positions[i] += system.positions[i] + (dt/6)*(k1_p + 2*k2_p + 2*k3_p + k4_p)
            new_velocities[i] += v + (dt / 6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
        return System(system.boides, new_positions, new_velocities, system.masses)'''

class Verlet:
    def __init__(self):
        self.prev_accel = None

    def step(self, system, dt):
        #cisto da ne bude greska pisem ovu liniju
        bodies = system
