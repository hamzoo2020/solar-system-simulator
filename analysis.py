import numpy as np
from constants import *
import matplotlib.pyplot as plt
from integrator import Euler, RK4, Verlet

def total_energy(system):
    Ek = 0.0
    Ep = 0.0
    for i in range(len(system.bodies)):
        velocity = np.linalg.norm(system.velocities[i])
        Ek += system.masses[i]*velocity**2/2
    for i in range(len(system.bodies)):
        for j in range(i + 1, len(system.bodies)):
            r_ij = np.linalg.norm(system.positions[i] - system.positions[j])
            Ep += -G*system.masses[i]*system.masses[j]/r_ij
    E = Ep + Ek
    return E

def run(system, method, dt, n_steps):
    energies = [total_energy(system)]
    times = [0]
    current = system
    for i in range(n_steps):
        times.append((i + 1)*dt)
        current = method.step(current, dt)
        energies.append(total_energy(current))
    return times, energies

def plot_total_energy(times, energies, name):
    x = times
    y = [energy/energies[0] for energy in energies]
    c = ''
    if name == "Euler": c = 'r'
    elif name == "RK4": c = 'b'
    elif name == "Verlet": c = 'g'
    plt.plot(x, y, c, label = name)
    plt.title(f"Graph of deviation of energy over time")
    plt.xlabel("dt[s]")
    plt.ylabel("E[t]/E[0]")
    plt.legend()
    plt.grid(True)


def convergence(system, method, dt_list, T_total, ref_dt):
    # Referenca — ista logika, da bude egzaktno na T_total
    n_ref = int(T_total / ref_dt)
    state_ref = system
    rk4 = RK4()
    for _ in range(n_ref):
        state_ref = rk4.step(state_ref, ref_dt)
    rem_ref = T_total - n_ref * ref_dt
    if rem_ref > 1e-9:
        state_ref = rk4.step(state_ref, rem_ref)
    ref_positions = state_ref.positions

    errors = []
    for dt in dt_list:
        n = int(T_total / dt)
        current = system
        for _ in range(n):
            current = method.step(current, dt)
        # Poslednji delimični korak — da se stigne TAČNO do T_total
        rem = T_total - n * dt
        if rem > 1e-9:
            current = method.step(current, rem)
        errors.append(np.linalg.norm(current.positions - ref_positions))
    return errors


def plot_converges(series):
    x = series[0][0]
    y1 = series[0][1]
    y2 = series[1][1]
    y3 = series[2][1]
    plt.loglog(x, y1, 'r', label = series[0][2])
    plt.loglog(x, y2, 'b', label = series[1][2])
    plt.loglog(x, y3, 'g', label = series[2][2])
    plt.title(f"Graph of convergence")
    plt.xlabel("dt[s]")
    plt.ylabel("Error[m]")
    plt.grid(True)
    plt.legend()
    plt.show()

def numerical_velocity(system, method, dt, n_steps):
    positions_list = []
    current = system
    for i in range(n_steps):
        current = method.step(current, dt)
        positions_list.append(current.positions)
    velocities = []
    for i in range(1, n_steps - 1):
        velocities.append((positions_list[i + 1] - positions_list[i - 1]) / (2*dt))
    return velocities

def numerical_acceleration(system, method, dt, n_steps):
    positions_list = []
    current = system
    for i in range(n_steps):
        current = method.step(current, dt)
        positions_list.append(current.positions)
    accelerations = []
    for i in range(1, n_steps - 1):
        accelerations.append((positions_list[i + 1] - 2*positions_list[i] + positions_list[i - 1]) / dt**2)
    return accelerations

def differentiation_error(system, method, dt, n_steps):
    int_vel = []
    int_pos = []
    current = system
    for i in range(n_steps):
        current = method.step(current, dt)
        int_vel.append(current.velocities.copy())
        int_pos.append(current.positions.copy())

    err_vel = []
    err_acc = []
    for i in range(1, n_steps - 1):
        v_int_i = int_vel[i]                    
        a_int_i = system.accelerations(int_pos[i])

        vel_num = (int_pos[i+1] - int_pos[i-1]) / (2*dt)
        acc_num = (int_pos[i+1] - 2*int_pos[i] + int_pos[i-1]) / dt**2

        vn = np.linalg.norm(v_int_i - vel_num)
        an = np.linalg.norm(a_int_i - acc_num)
        err_vel.append(vn)
        err_acc.append(an)
    return err_vel, err_acc

def plot_differentiation_error(system, method, dt, n_steps):
    err_vel, err_acc = differentiation_error(system, method, dt, n_steps)

    x = [dt*i for i in range(1, n_steps - 1)]
    plt.plot(x, err_vel, 'r', label = "Velocity differentiation error")
    plt.title("Differentiation error of velocity")
    plt.xlabel("t[s]")
    plt.ylabel("dV[m/s]")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(x, err_acc, 'r', label = "Acceleration differentiation error")
    plt.title("Differentiation error of acceleration")
    plt.xlabel("t[s]")
    plt.ylabel("da[m/s^2]")
    plt.legend()
    plt.grid(True)
    plt.show()













