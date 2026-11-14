"""
Prikaz grafika za proveru i izveštaj.
Pokreće se: python tests_manual.py
Nema assert-a — samo crta grafike.
"""

import numpy as np
import matplotlib.pyplot as plt

from celestial_object import CelestialObject
from constants import *
from system import System
from integrator import Euler, RK4, Verlet
from analysis import run, convergence, plot_total_energy, plot_converges


def make_sun_earth():
    sun = CelestialObject("Sun", SUN_MASS, SUN_RADIUS)
    earth = CelestialObject("Earth", EARTH_MASS, EARTH_RADIUS)
    bodies = [sun, earth]
    positions = np.array([SUN_POSITIONS, EARTH_POSITIONS])
    velocities = np.array([SUN_VELOCITY, EARTH_VELOCITY])
    return System(bodies, positions, velocities)


def show_energy_plot():
    """
    Grafik 1: relativna promena energije kroz vreme za sve tri metode.
    Očekivano: Euler raste, RK4 sporo raste, Verlet osciluje oko 1.
    """
    print("=== Grafik 1: energija ===")
    system = make_sun_earth()
    dt = 3600.0
    n_steps = 1000   # ~41 dan

    t_e, e_e = run(system, Euler(),  dt, n_steps)
    t_r, e_r = run(system, RK4(),    dt, n_steps)
    t_v, e_v = run(system, Verlet(), dt, n_steps)

    plt.figure(figsize=(10, 6))
    plot_total_energy(t_e, e_e, "Euler")
    plot_total_energy(t_r, e_r, "RK4")
    plot_total_energy(t_v, e_v, "Verlet")
    plt.yscale("symlog", linthresh=1e-10)   # vidi i oscilacije i drift
    plt.savefig("energy.png", dpi=150, bbox_inches="tight")
    plt.show()


def show_convergence_plot():
    """
    Grafik 2: konvergencija (log-log) za sve tri metode.
    Očekivano: Euler nagib ~1, Verlet ~1.5-2, RK4 ~4.
    """
    print("=== Grafik 2: konvergencija ===")
    system = make_sun_earth()
    dt_list = [7200.0, 3600.0, 1800.0, 900.0, 450.0]
    T_total = 3_600_000.0   # ~41 dan — deljivo sa svim dt
    ref_dt = 10.0

    err_e = convergence(system, Euler(),  dt_list, T_total, ref_dt)
    err_r = convergence(system, RK4(),    dt_list, T_total, ref_dt)
    err_v = convergence(system, Verlet(), dt_list, T_total, ref_dt)

    # Ispiši nagibe (korisno za odbranu)
    for name, errs in [("Euler", err_e), ("RK4", err_r), ("Verlet", err_v)]:
        slope = np.polyfit(np.log(dt_list), np.log(errs), 1)[0]
        print(f"{name:>7}: nagib = {slope:.3f}")

    plt.figure(figsize=(10, 6))
    plot_converges([
        (dt_list, err_e, "Euler"),
        (dt_list, err_r, "RK4"),
        (dt_list, err_v, "Verlet"),
    ])
    # plot_converges poziva plt.show() — sačuvaj pre toga ako hoćeš
    # (vidi napomenu ispod)


if __name__ == "__main__":
    show_energy_plot()
    show_convergence_plot()