import numpy as np
import matplotlib.pyplot as plt
from constants import *
from integrator import Euler, RK4, Verlet
import os

def plot_orbits_2d(system, method, dt, n_steps):
    positions_history = [system.positions]
    current = system
    for i in range(n_steps):
        current = method.step(current, dt)
        positions_history.append(current.positions.copy())

    plt.figure(figsize=(10, 10))
    colors = ['gray', 'orange', 'yellow', 'blue', 'red', 'brown', 'gold', 'cyan', 'navy']
    for i in range(len(system.bodies)):
        body = system.bodies[i]
        if i == 0:
            plt.plot(0, 0, 'o', color='yellow', markersize=5, label="Sun")
            continue
        xs = [pos[i][0] / AU for pos in positions_history]
        ys = [pos[i][1] / AU for pos in positions_history]
        xt = xs[len(xs) - 1]
        yt = ys[len(ys) - 1]
        plt.plot(xs, ys, color=colors[i], label=body.name + " path", linewidth=1) 
        plt.plot(xt, yt, 'o', color=colors[i], markersize=3, label=body.name)

    plt.xlabel("x[AU]")
    plt.ylabel("y[AU]")
    plt.title("2D vizualization")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.show()
    plt.close()

def run_3d_visualization(system, method, dt, n_steps_per_frame=1):
    from vpython import canvas, sphere, vector, color, rate, button, slider, text
    from analysis import (
        run, convergence, differentiation_error, 
        plot_total_energy, plot_converges, plot_differentiation_error
    )
    
    state = {
        "running": True,
        "speed": 1.0,
        "current_system": system
    }

    AU_TO_VPY = 100 
    
    scene = canvas(title="Solar System Simulator", width=1200, height=700, background=color.black, align='left')
    scene.camera.pos = vector(0, -3000, 1500)
    scene.camera.axis = vector(0, 3000, -1500)

    colors = [
        vector(1.0, 1.0, 0.0), vector(0.5, 0.5, 0.5), vector(1.0, 0.6, 0.2), 
        vector(0.2, 0.4, 1.0), vector(1.0, 0.2, 0.2), vector(0.8, 0.5, 0.2), 
        vector(0.9, 0.8, 0.5), vector(0.5, 0.8, 0.9), vector(0.2, 0.3, 0.8)
    ]
    
    vpy_bodies = []
    for i, body in enumerate(system.bodies):
        r_display = 20 if i == 0 else body.radius * 1e-6 * 2 
        s = sphere(pos=vector(0,0,0), radius=r_display, color=colors[i], make_trail=(i > 0), retain=50, trail_radius=0.3)
        vpy_bodies.append(s)

    for i in range(len(system.bodies)):
        pos = system.positions[i] / AU * AU_TO_VPY
        vpy_bodies[i].pos = vector(pos[0], pos[1], pos[2])

    def toggle_play(b):
        state["running"] = not state["running"]
        b.text = "Play" if not state["running"] else "Pause"

    def change_speed(s):
        state["speed"] = s.value

    def show_energy(b):
        state["running"] = False
        btn_play.text = "Play"
        
        t, e = run(state["current_system"], method, dt, 500)
        plot_total_energy(t, e, "Energija")

    def show_convergence(b):
        state["running"] = False
        btn_play.text = "Play"
        
        dt_list = [3600.0, 1800.0, 900.0]
        T_total = 10000.0
        ref_dt = 100.0      
        
        err_e = convergence(state["current_system"], Euler(), dt_list, T_total, ref_dt)
        err_r = convergence(state["current_system"], RK4(),   dt_list, T_total, ref_dt)
        err_v = convergence(state["current_system"], Verlet(),dt_list, T_total, ref_dt)
        
        plot_converges([(dt_list, err_e, "Euler"), (dt_list, err_r, "RK4"),(dt_list, err_v, "Verlet")])

    def show_diff_error(b):
        state["running"] = False
        btn_play.text = "Play"
        
        err_v, err_a = differentiation_error(state["current_system"], method, dt, 500)
        plot_differentiation_error(state["current_system"], method, dt, 500)


    scene.append_to_caption("\n\n") 
    
    btn_play = button(text="Pause", bind=toggle_play)
    scene.append_to_caption("   Brzina: ")
    sl = slider(min=0.1, max=20.0, value=1.0, step=0.1, bind=change_speed, length=200)
    
    scene.append_to_caption("\n\n")
    btn_energy = button(text="Grafik Energije", bind=show_energy)
    scene.append_to_caption("  ") 
    btn_conv = button(text="Grafik Konvergencije", bind=show_convergence)
    scene.append_to_caption("  ") 
    btn_diff = button(text="Greska Diferenciranja", bind=show_diff_error)


    while True:
        rate(100)
        
        if not state["running"]:
            continue
            
        steps_to_take = int(n_steps_per_frame * state["speed"])
        
        if steps_to_take < 1:
            steps_to_take = 1
            
        for i in range(steps_to_take):
            state["current_system"] = method.step(state["current_system"], dt)
            
        for i in range(len(system.bodies)):
            pos = state["current_system"].positions[i] / AU * AU_TO_VPY
            vpy_bodies[i].pos = vector(pos[0], pos[1], pos[2])
