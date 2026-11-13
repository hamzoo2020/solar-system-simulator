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

class Verlet:
    def __init__(self):
        self.prev_accel = None

    def step(self, system, dt):
        self.prev_accel = system.accelerations()
        new_positions = system.positions + system.velocities*dt + (1/2)*self.prev_accel*dt**2
        new_accelerations = system.accelerations(new_positions)
        new_velocities = system.velocities + (self.prev_accel + new_accelerations)/2 * dt
        return System(system.bodies, new_positions, new_velocities)
