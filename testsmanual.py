"""
Provera da graficke funkcije rade bez greske.
Pokreni: python tests_manual.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from celestial_object import CelestialObject
from constants import *
from system import System
from integrator import Euler, RK4, Verlet
from analysis import (
    run, total_energy, convergence,
    differentiation_error,
    plot_total_energy, plot_converges, plot_differentiation_error, plot_error_vs_h
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

def test_error_vs_h_plot():
    """
    Provera da plot_error_vs_h radi bez greske i da podaci imaju smisla.
    Ocekivano: 'V' oblik — greska opada O(h^2) za velike h, raste za male (round-off).
    """
    print("=== Grafik: error vs h ===")
    system = make_sun_earth()
    method = RK4()
    dt_list = [60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0]
    n_steps = 500   # kratko, samo da provera prodje brzo

    # 1. Pozovi differentiation_error za svaki dt — proveri da vraca smislene vrednosti
    err_v_mean = []
    err_a_mean = []
    for dt in dt_list:
        err_v, err_a = differentiation_error(system, method, dt, n_steps)
        assert len(err_v) > 0, f"dt={dt}: prazan err_v"
        assert len(err_a) > 0, f"dt={dt}: prazan err_a"
        assert all(np.isfinite(v) for v in err_v), f"dt={dt}: err_v ima NaN/inf"
        assert all(np.isfinite(a) for a in err_a), f"dt={dt}: err_a ima NaN/inf"
        err_v_mean.append(np.mean(err_v))
        err_a_mean.append(np.mean(err_a))
        print(f"  dt={dt:>6.0f}  mean(err_v)={np.mean(err_v):.3e}  mean(err_a)={np.mean(err_a):.3e}")

    # 2. Proveri da su srednje greske pozitivne
    assert all(e > 0 for e in err_v_mean), "err_v_mean mora biti pozitivno"
    assert all(e > 0 for e in err_a_mean), "err_a_mean mora biti pozitivno"

    # 3. Nagib za velike h (gde O(h^2) dominira)
    mask = np.array(dt_list) > 200
    try:
        k_v = np.polyfit(np.log(np.array(dt_list)[mask]),
                         np.log(np.array(err_v_mean)[mask]), 1)[0]
        k_a = np.polyfit(np.log(np.array(dt_list)[mask]),
                         np.log(np.array(err_a_mean)[mask]), 1)[0]
        print(f"  nagib (v, velike h) = {k_v:.3f}")
        print(f"  nagib (a, velike h) = {k_a:.3f}")
    except Exception as e:
        print(f"  polyfit preskocen: {e}")

    # 4. Pozovi plot funkciju — proveri da ne puca
    plot_error_vs_h(system, method, dt_list, n_steps)
    plt.close("all")

    print("  grafici prikazani (bez savefig unutar funkcije)\n")

if __name__ == "__main__":
    print("Provera grafika...\n")
    test_error_vs_h_plot()
    print("Svi grafici prosli.")