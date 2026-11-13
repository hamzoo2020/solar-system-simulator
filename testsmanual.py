import numpy as np
from celestial_object import CelestialObject
from constants import *
from system import System
from integrator import Euler, RK4, Verlet
from analysis import run, total_energy, plot_total_energy
import matplotlib.pyplot as plt


def make_sun_earth():
    sun = CelestialObject("Sun", SUN_MASS, SUN_RADIUS)
    earth = CelestialObject("Earth", EARTH_MASS, EARTH_RADIUS)
    bodies = [sun, earth]
    positions = np.array([SUN_POSITIONS, EARTH_POSITIONS])
    velocities = np.array([SUN_VELOCITY, EARTH_VELOCITY])
    return System(bodies, positions, velocities)


def test_run_structure():
    """
    run() mora da vrati (times, energies) iste dužine,
    times[0] = 0, i energies[0] = total_energy(initial).
    """
    system = make_sun_earth()
    times, energies = run(system, Euler(), 3600.0, 10)

    print("=== Test: struktura run() ===")
    assert len(times) == len(energies) == 11, \
        f"treba 11 elemenata (n_steps+1), dobijeno {len(times)}, {len(energies)}"
    assert times[0] == 0, "times[0] mora biti 0"
    assert abs(times[1] - 3600.0) < 1e-9, "times[1] mora biti dt"
    assert abs(energies[0] - total_energy(system)) < 1e-6, \
        "energies[0] mora biti E(initial)"

    print(f"len(times) = {len(times)}, times[0] = {times[0]}, E(0) = {energies[0]:.3e}")
    print("OK\n")


def test_euler_energy_drifts():
    """
    Euler mora da driftuje energiju — |E(1000)/E(0) - 1| > 1e-4.
    """
    system = make_sun_earth()
    times, energies = run(system, Euler(), 3600.0, 1000)

    rel = abs(energies[-1] / energies[0] - 1)
    print("=== Test: Euler driftuje ===")
    print(f"|E(1000)/E(0) - 1| = {rel:.3e}")
    assert rel > 1e-4, f"Euler mora da driftuje, dobijeno {rel:.3e}"

    print("OK\n")


def test_verlet_energy_stable():
    """
    Verlet ne sme da driftuje — |E(1000)/E(0) - 1| < 1e-3.
    (Osciluje, ali ne raste sistematski.)
    """
    system = make_sun_earth()
    times, energies = run(system, Verlet(), 3600.0, 1000)

    rel = abs(energies[-1] / energies[0] - 1)
    print("=== Test: Verlet ne driftuje ===")
    print(f"|E(1000)/E(0) - 1| = {rel:.3e}")
    assert rel < 1e-3, f"Verlet ne sme da driftuje, dobijeno {rel:.3e}"

    print("OK\n")


def test_verlet_better_than_euler():
    """
    Posle 1000 koraka, Verlet mora imati manji drift od Euler-a.
    """
    system = make_sun_earth()

    _, e_euler = run(system, Euler(), 3600.0, 1000)
    _, e_verlet = run(system, Verlet(), 3600.0, 1000)

    drift_e = abs(e_euler[-1] / e_euler[0] - 1)
    drift_v = abs(e_verlet[-1] / e_verlet[0] - 1)

    print("=== Test: Verlet bolji od Euler ===")
    print(f"Euler drift:  {drift_e:.3e}")
    print(f"Verlet drift: {drift_v:.3e}")
    assert drift_v < drift_e, "Verlet mora imati manji drift"

    print("OK\n")

def test():
    system = make_sun_earth()
    n_steps = 1000
    dt = 3600.0

    t_e, e_e = run(system, Euler(),  dt, n_steps)
    t_r, e_r = run(system, RK4(),    dt, n_steps)
    t_v, e_v = run(system, Verlet(), dt, n_steps)

    plot_total_energy(t_e, e_e, "Euler")
    plot_total_energy(t_r, e_r, "RK4")
    plot_total_energy(t_v, e_v, "Verlet")

    plt.yscale("symlog", linthresh=1e-10)
    plt.show()

if __name__ == "__main__":
    print("Pokrećem testove za energiju...\n")
    test_run_structure()
    test_euler_energy_drifts()
    test_verlet_energy_stable()
    test_verlet_better_than_euler()
    print("Svi testovi prošli.")
    test()