from celestial_object import CelestialObject
from constants import *
from system import System
from integrator import Euler, RK4, Verlet
import numpy as np

def build_solar_system():
    bodies = [
        CelestialObject("Sun", SUN_MASS, SUN_RADIUS),
        CelestialObject("Mercury", MERCURY_MASS, MERCURY_RADIUS),
        CelestialObject("Venus", VENUS_MASS, VENUS_RADIUS),
        CelestialObject("Earth", EARTH_MASS, EARTH_RADIUS),
        CelestialObject("Mars", MARS_MASS, MARS_RADIUS),
        CelestialObject("Jupiter", JUPITER_MASS, JUPITER_RADIUS),
        CelestialObject("Saturn", SATURN_MASS, SATURN_RADIUS),
        CelestialObject("Uranus", URANUS_MASS, URANUS_RADIUS),
        CelestialObject("Neptune", NEPTUNE_MASS, NEPTUNE_RADIUS)
    ]
    positions = np.array([SUN_POSITIONS, MERCURY_POSITIONS, VENUS_POSITIONS, EARTH_POSITIONS, MARS_POSITIONS, JUPITER_POSITIONS, SATURN_POSITIONS, URANUS_POSITIONS, NEPTUNE_POSITIONS])
    velocities = np.array([
        SUN_VELOCITY,
        MERCURY_VELOCITY,
        VENUS_VELOCITY,
        EARTH_VELOCITY,
        MARS_VELOCITY,
        JUPITER_VELOCITY,
        SATURN_VELOCITY,
        URANUS_VELOCITY,
        NEPTUNE_VELOCITY
    ])
    return System(bodies, positions, velocities)
def main():
    initial_system = build_solar_system()
    rk4 = RK4()
    dt = 3600.0
    n_steps = int(86400 / dt)   # 24 koraka = 1 dan
    current = initial_system
    for _ in range(n_steps):
        current = rk4.step(current, dt)

    print("\n=== Posle 1 dana ===")
    for i, body in enumerate(current.bodies):
        r0 = np.linalg.norm(initial_system.positions[i])
        r1 = np.linalg.norm(current.positions[i])
        drift = abs(r1 - r0) / r0 * 100
        print(f"{body.name:>8}: |r| = {r1/AU:.3f} AU, drift = {drift:.6f}%")

if __name__ == '__main__':
    main() 