import numpy as np
from celestial_object import CelestialObject
from constants import *
from system import System
from integrator import Euler, RK4, Verlet

def main():
    earth = CelestialObject("Earth", EARTH_MASS, EARTH_RADIUS)
    sun = CelestialObject("Sun", SUN_MASS, SUN_RADIUS)

    bodies = [sun, earth]   # Sunce na indeksu 0, Zemlja na 1
    positions = np.array([SUN_POSITIONS, EARTH_POSITIONS])
    velocities = np.array([SUN_VELOCITY, EARTH_VELOCITY])

    system = System(bodies, positions, velocities)
    euler = Euler()
    rk4 = RK4()

    '''dt = 3600.0
    new_system = euler.step(system, dt)
    new_system4 = rk4.step(system, 3600)

    diff_pos = np.linalg.norm(new_system4.positions[1] - new_system.positions[1])
    print(f"|Δr| (RK4 vs Euler) = {diff_pos:.3e} m")
    assert 1e2 < diff_pos < 1e7, "razlika van očekivanog opsega"'''
    pos_before = system.positions.copy()
    vel_before = system.velocities.copy()

    verlet = Verlet()
    new_system = verlet.step(system, 3600.0)

    assert np.array_equal(system.positions, pos_before), "mutira positions!"
    assert np.array_equal(system.velocities, vel_before), "mutira velocities!"
    assert not np.array_equal(new_system.positions, system.positions), "ne menja ništa!"

    print("Test A prošao: immutable, i step menja stanje")

if __name__ == "__main__":
    main()