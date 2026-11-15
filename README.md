# Solar System Numerical Simulator

## Project Description

**Solar System Numerical Simulator** is an application for simulating the motion of the planets in the Solar System. Unlike standard animations that follow predefined paths, the movement of the planets is calculated in real time by solving the N-body problem.

The simulator uses Newton's law of universal gravitation to calculate the mutual interaction between the Sun and the eight planets. The mathematical model is a system of second-order differential equations, solved numerically with three different methods: Euler, 4th-order Runge-Kutta (RK4), and Verlet.

The application features an interactive 3D visualization, time-control features, and a module for numerical analysis of errors and energy conservation.

## Key Features

- **3D Visualization:** Display of the Sun and eight planets with realistic relative distances.
- **Interactive Control Panel:**
  - Play/Pause button for simulation control.
  - Speed slider to accelerate or slow down the flow of time (0.1x to 20x).
- **Numerical Analysis:** Buttons to generate plots for:
  - Relative change of the system's total energy over time.
  - Convergence (order of accuracy) of the numerical methods.
  - Numerical differentiation errors (first and second derivatives).
- **Selectable Numerical Method:** The simulation can be run with Euler, RK4, or Verlet by passing the chosen integrator to the visualization function in `main.py`.
- **Custom Textures (Optional):** Solid colors can be replaced with NASA-based planet textures loaded into the VPython spheres (see [Textures](#textures)).

## Numerical Methods and Precision

Three numerical methods for solving differential equations were implemented independently (without built-in libraries such as `scipy.integrate`):

- **Euler Method:** The simplest method, order of accuracy 1. Used as a baseline for comparison.
- **Runge-Kutta 4th Order (RK4):** Method of order 4. It gives the smallest error over short time intervals.
- **Verlet Method:** A symplectic integrator of order 2. Although formally of a lower order than RK4, it conserves the system's energy best over long time intervals, which is demonstrated by the analysis in this project.

The methods were verified through convergence analysis (log-log plots), where the error decreases with the time step `dt` at the rate predicted by theory.

## Project Structure

The project is modular and separates the physical model, numerical methods, and the user interface:

| File | Description |
|------|-------------|
| `main.py` | Entry point of the program. Connects all modules and starts the simulation. |
| `constants.py` | Initial conditions (positions, velocities, masses) from NASA JPL Horizons, plus the gravitational constant and astronomical unit. |
| `celestial_object.py` | Class that stores static data about the bodies (name, mass, radius). |
| `system.py` | `System` class that stores the current state (positions and velocities) and contains `accelerations()`, which computes gravitational acceleration for all bodies (N-body model). |
| `integrator.py` | Euler, RK4, and Verlet classes with a common `step()` method. |
| `analysis.py` | Functions for numerical analysis (energy, convergence, numerical differentiation, and plotting). |
| `visualization.py` | Functions for 2D orbit plotting (Matplotlib) and the interactive 3D animation (VPython). |

## Installation and Usage

### Prerequisites

Python 3.8+ is required. Using a virtual environment is highly recommended.

### 1. Clone the project

```bash
git clone https://github.com/your-username/solar-system-simulator.git
cd solar-system-simulator
```

### 2. Create a virtual environment and install dependencies

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If PowerShell blocks the activation script, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

If there is no `requirements.txt`, install the libraries manually:

```bash
pip install numpy matplotlib vpython
```

### 3. Run the simulation

```bash
python main.py
```

A web browser (or a VPython window) will open, showing the 3D view of the Solar System with the control panel below the scene.

## Textures

To use planet textures:

1. Download equirectangular (2:1) texture maps, for example from [Solar System Scope](https://www.solarsystemscope.com/textures/) or NASA's Solar System resources.
2. Place them in a `textures/` folder next to `main.py`.
3. Pass them to the spheres in `vizualization.py` using `texture="textures/earth.jpg"` and `color=color.white`.

## Data Source

The initial conditions (positions and velocities of the planets) were obtained from the [NASA JPL Horizons](https://ssd.jpl.nasa.gov/horizons/) system for the epoch November 11, 2026. A heliocentric reference frame (ecliptic J2000) was used, with units converted to SI (meters, seconds, kilograms).

## Technologies

- **Python:** main programming language.
- **NumPy:** vector operations and numerical computations.
- **VPython:** 3D WebGL visualization and interactive controls.
- **Matplotlib:** 2D numerical analysis plots.

## Note on Accuracy

The simulator uses Newtonian gravity, excluding relativistic corrections and the influence of asteroids. While the orbits are stable and visually accurate, the project is intended to demonstrate numerical methods rather than precise astronomical navigation.
