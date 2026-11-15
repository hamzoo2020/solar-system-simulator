from celestial_object import CelestialObject
from constants import *
from system import System
from integrator import Euler, RK4, Verlet
import numpy as np
from vizualization import (plot_orbits_2d, run_3d_visualization)

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
    '''plot_orbits_2d(initial_system, verlet, dt, 40000)''' #idelan odnos brzine izvrsavanja i preglednosti n = 40000, za n = 400000 treba jedno 10min da se ucita
    initial_system = build_solar_system()
    rk4 = RK4()
    euler = Euler()
    verlet = Verlet()
    dt = 3600.00
    #plot_orbits_2d(initial_system, rk4, dt, 4000)
    run_3d_visualization(initial_system, verlet, dt, n_steps_per_frame=10)

if __name__ == '__main__':
    main() 