import numpy as np
from celestial_object import CelestialObject
from constants import *
from system import System
from integrator import Euler, RK4

def main():
    earth = CelestialObject("Earth", EARTH_MASS, EARTH_RADIUS)
    sun = CelestialObject("Sun", SUN_MASS, SUN_RADIUS)

    bodies = [sun, earth]   # Sunce na indeksu 0, Zemlja na 1
    positions = np.array([SUN_POSITIONS, EARTH_POSITIONS])
    velocities = np.array([SUN_VELOCITY, EARTH_VELOCITY])

    system = System(bodies, positions, velocities)
    euler = Euler()
    rk4 = RK4()

    dt = 3600.0
    new_system = euler.step(system, dt)
    new_system4 = rk4.step(system, 3600)

    diff_pos = np.linalg.norm(new_system4.positions[1] - new_system.positions[1])
    print(f"|Δr| (RK4 vs Euler) = {diff_pos:.3e} m")
    assert 1e2 < diff_pos < 1e7, "razlika van očekivanog opsega"
    

if __name__ == "__main__":
    main()