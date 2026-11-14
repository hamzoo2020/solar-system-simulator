"""
Provera da graficke funkcije rade bez greske.
Pokreni: python tests_manual.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # bez GUI — samo savefig, brze i bez blokiranja
import matplotlib.pyplot as plt

from celestial_object import CelestialObject
from constants import *
from system import System
from integrator import Euler, RK4, Verlet
from analysis import (
    run, total_energy, convergence,
    differentiation_error,
    plot_total_energy, plot_converges, plot_differentiation_error,
)


def make_sun_earth():
    sun = CelestialObject("Sun", SUN_MASS, SUN_RADIUS)
    earth = CelestialObject("Earth", EARTH_MASS, EARTH_RADIUS)
    bodies = [sun, earth]
    positions = np.array([SUN_POSITIONS, EARTH_POSITIONS])
    velocities = np.array([SUN_VELOCITY, EARTH_VELOCITY])
    return System(bodies, positions, velocities)


def check_sane(values, name):
    """Provera da niz ima smislene vrednosti."""
    assert len(values) > 0, f"{name}: prazan niz"
    assert all(np.isfinite(v) for v in values), f"{name}: ima NaN/inf"
    print(f"  {name}: len={len(values)}, "
          f"min={min(values):.3e}, max={max(values):.3e}")


def test_energy_plot():
    print("=== Grafik: energija ===")
    system = make_sun_earth()
    dt = 3600.0
    n_steps = 500

    t_e, e_e = run(system, Euler(),  dt, n_steps)
    t_r, e_r = run(system, RK4(),    dt, n_steps)
    t_v, e_v = run(system, Verlet(), dt, n_steps)

    check_sane(e_e, "energies Euler")
    check_sane(e_r, "energies RK4")
    check_sane(e_v, "energies Verlet")

    plt.figure(figsize=(10, 6))
    plot_total_energy(t_e, e_e, "Euler")
    plot_total_energy(t_r, e_r, "RK4")
    plot_total_energy(t_v, e_v, "Verlet")
    plt.savefig("test_energy.png", dpi=100)
    plt.close()
    print("  sacuvano test_energy.png\n")


def test_convergence_plot():
    print("=== Grafik: konvergencija ===")
    system = make_sun_earth()
    dt_list = [7200.0, 3600.0, 1800.0, 900.0]
    T_total = 1_000_000.0   # kratko, samo da provera prodje brzo
    ref_dt = 50.0

    err_e = convergence(system, Euler(),  dt_list, T_total, ref_dt)
    err_r = convergence(system, RK4(),    dt_list, T_total, ref_dt)
    err_v = convergence(system, Verlet(), dt_list, T_total, ref_dt)

    check_sane(err_e, "conv Euler")
    check_sane(err_r, "conv RK4")
    check_sane(err_v, "conv Verlet")

    plt.figure(figsize=(10, 6))
    plot_converges([
        (dt_list, err_e, "Euler"),
        (dt_list, err_r, "RK4"),
        (dt_list, err_v, "Verlet"),
    ])
    plt.close()   # plot_converges ima plt.show(); sa Agg ne blokira
    print("  sacuvano convergence.png (preko plot_converges)\n")


def test_differentiation_plot():
    print("=== Grafik: numericko diferenciranje ===")
    system = make_sun_earth()
    dt = 60.0
    n_steps = 1000

    err_v, err_a = differentiation_error(system, RK4(), dt, n_steps)
    check_sane(err_v, "diff err vel")
    check_sane(err_a, "diff err acc")

    plot_differentiation_error(system, RK4(), dt, n_steps)
    plt.close("all")
    print("  differentiation grafici prikazani (bez savefig, samo plt.show)\n")


if __name__ == "__main__":
    print("Provera grafika...\n")
    test_energy_plot()
    test_convergence_plot()
    test_differentiation_plot()
    print("Svi grafici prosli.")