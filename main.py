from celestial_object import CelestialObject
from constants import *
from system import System
from integrator import Euler
import numpy as np

def main():
    earth = CelestialObject("Earth", EARTH_MASS, EARTH_RADIUS)
    sun = CelestialObject("Sun", SUN_MASS, SUN_RADIUS)
    bodies = [earth, sun]
    velocities = np.array([EARTH_VELOCITY, SUN_VELOCITY])
    positions = np.array([EARTH_POSITIONS, SUN_POSITIONS])
    initial_system = System(bodies, positions, velocities)
    euler = Euler()
    pos_before = initial_system.positions.copy()
    new_system = euler.step(initial_system, 3600)
    assert np.array_equal(initial_system.positions, pos_before), "step mutira ulaz!"

if __name__ == '__main__':
    main() 