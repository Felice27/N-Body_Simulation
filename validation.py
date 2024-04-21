"""
File: validation.py

Description: This file contains known scenarios that can be used to validate the N-Body Simulation.  We will look at 2 masses orbiting their center of mass, 3 masses in an Euler Triangle, and 3 masses in a figure-8 (colinear) orbit.

Author: Jack Felice

Revision History:
    2024-04-20: Initial creation and upload to repository
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import ffmpeg # Writer for saving animations to .mp4 file.  Pylance and the like may think this variable is unused, but it is necessary for saving animations.
import os
import multiprocessing
from Definitions import *

# Define main loop so that multiprocessing can be used
if __name__ == '__main__':
    num_threads = os.cpu_count()
    print("Number of available threads:", num_threads),
    if num_threads == 1:
        print("Will not be able to use multithreading to speed up program.  Animation will not generate until figures have been closed.")
    else:
        print("Multithreading available.  Components will be parallellized where possible.")
    if not os.path.exists("Results/Validation"):
        os.makedirs("Results/Validation")

    subdir_twobody = "Validation/2-body"

    if os.path.exists(f"Results/{subdir_twobody}"):
        print("Directory already exists. Overwriting...")
    os.makedirs(f"Results/{subdir_twobody}", exist_ok=True)

    # Constants: to be used in all simulations

    G = 1.0 # Gravitational constant
    N = 10 # Number of bodies
    # Time 
    t_start = 0.0
    t_end = 30.0 # Increased time to allow for more orbits
    dt = 1e-2

    # Softening parameter
    eps = 0.0 # No softening for validation!  With eps = 0, the equations of motion become the ordinary Newtonian equations of motion.  This is useful for validating the simulation.

    # Always want to show and save all plots
    show_animation = 1
    show_energy = 1
    show_com = 1
    save_plots = 1
    trails = 1

    # First Validation: 2 masses orbiting their center of mass
    N = 2
    # Masses
    M = np.array([1.0, 1.0])
    # Initial positions
    R0 = np.array([[0.5, 0.0], [-0.5, 0.0]])
    # Initial velocities
    V0 = np.array([[0.0, 1./np.sqrt(2)], [0.0, -1./np.sqrt(2)]])
    class_args = (G, M, R0, V0, t_start, t_end, dt, eps)
    print("Running 2-body validation...")
    t0 = time.time()
    two_body = CompiledOrbitSolver(*class_args)
    R_twobody, V_twobody, KE_twobody, PE_twobody = two_body.solve()
    t1 = time.time()
    print("2-body validation completed in", t1-t0, "seconds.")
    solver_twobody = OrbitSolver(*class_args) # Uncompiled solver for plotting
    # Show and save the results
    if num_threads >= 2:
        # Reinitialize the solver to prevent issues with multithreading, as the compiled solver cannot be pickled (even though it's identical to the non-compiled solver).  I could potentially fix this by instead passing all the attributes of the solver to the function, but that would be a lot of work for little gain.
        # Create and start the threads
        show_thread = multiprocessing.Process(target=make_plots, args=(R_twobody, M, N, KE_twobody, PE_twobody, solver_twobody, dt, trails, show_animation, show_energy, show_com, save_plots, subdir_twobody))
        save_thread = multiprocessing.Process(target=save_anim, args=(R_twobody, M, N, solver_twobody, subdir_twobody, dt, trails, show_animation))
        show_thread.start()
        save_thread.start()
        #show_thread.join() # Uncomment to wait until the plots are shown before continuing
        #save_thread.join() # Uncomment to wait until the animation is saved before continuing
    else:
        # If only one thread available, run sequentially
        make_plots(R_twobody, M, N, KE_twobody, PE_twobody, solver_twobody, dt, trails, show_animation, show_energy , show_com, save_plots, subdir_twobody)
        save_anim(R_twobody, M, N, solver_twobody, subdir_twobody, dt, trails, show_animation)
    
    # Second Validation: 3 masses in an Euler Triangle
    N = 3
    # Masses
    M = np.array([1.0, 1.0, 1.0])
    # Initial positions
    theta = 2*np.pi/3
    R0 = np.array([[1.0, 0], [np.cos(theta), np.sin(theta)], [np.cos(2*theta), np.sin(2*theta)]]) # Euler Triangle
    # Initial velocities
    V0 = np.sqrt(1.0/np.sqrt(3))*np.array([[0, 1.0], [-np.sqrt(3)/2, -0.5], [np.sqrt(3)/2, -1/2]]) # Magnitude from equations of motion, direction from trigonometry
    dt = 1e-2
    class_args = (G, M, R0, V0, t_start, t_end, dt, eps)
    print("Running Euler Triangle validation...")
    t0 = time.time()
    euler_triangle = CompiledOrbitSolver(*class_args)
    R_euler, V_euler, KE_euler, PE_euler = euler_triangle.solve()
    t1 = time.time()
    print("Euler Triangle validation completed in", t1-t0, "seconds.")
    solver = OrbitSolver(*class_args) # Uncompiled solver for plotting
    # Show and save the results
    subdir_euler = "Validation/Euler"
    if os.path.exists(f"Results/{subdir_euler}"):
        print("Directory already exists. Overwriting...")
    os.makedirs(f"Results/{subdir_euler}", exist_ok=True)

    euler_solver = OrbitSolver(*class_args) # Uncompiled solver for plotting
    
    if num_threads >= 2:
        # Create and start the threads
        show_thread = multiprocessing.Process(target=make_plots, args=(R_euler, M, N, KE_euler, PE_euler, euler_solver, dt, trails, show_animation, show_energy, show_com, save_plots, subdir_euler))
        save_thread = multiprocessing.Process(target=save_anim, args=(R_euler, M, N, euler_solver, subdir_euler, dt, trails, show_animation))
        show_thread.start()
        save_thread.start()
        #show_thread.join() # Uncomment to wait until the plots are shown before continuing
        #save_thread.join() # Uncomment to wait until the animation is saved before continuing
    else:
        # If only one thread available, run sequentially
        make_plots(R_euler, M, N, KE_euler, PE_euler, euler_solver, dt, trails, show_animation, show_energy , show_com, save_plots, subdir_euler)
        save_anim(R_euler, M, N, euler_solver, subdir_euler, dt, trails, show_animation)

    # Third Validation: 3 masses in a figure-8 (colinear) orbit
    N = 3
    # Masses
    M = np.array([1.0, 1.0, 1.0])
    # Initial positions
    R0 = np.array([[0.97000436, -0.24308753], [-0.97000436, 0.24308753], [0., 0.]]) # Colinear
    # Initial velocities
    V3 = np.array([-0.932407370, -0.8647314600]) # Initial conditions from Chenciner and Montgomery (2000)
    V0 = np.array([-0.5 * V3, -0.5 * V3, V3])
    dt = 1e-2
    class_args = (G, M, R0, V0, t_start, t_end, dt, eps)
    print("Running figure-8 validation...")
    t0 = time.time()
    figure_eight = CompiledOrbitSolver(*class_args)
    R_figure, V_figure, KE_figure, PE_figure = figure_eight.solve()
    t1 = time.time()
    print("Figure-8 validation completed in", t1-t0, "seconds.")
    # Show and save the results
    subdir_figure_eight = "Validation/Figure-8"
    if os.path.exists(f"Results/{subdir_figure_eight}"):
        print("Directory already exists. Overwriting...")
    os.makedirs(f"Results/{subdir_figure_eight}", exist_ok=True)

    figure_eight_solver = OrbitSolver(*class_args) # Uncompiled solver for plotting

    if num_threads >= 2:
        # Create and start the threads
        show_thread = multiprocessing.Process(target=make_plots, args=(R_figure, M, N, KE_figure, PE_figure, figure_eight_solver, dt, trails, show_animation, show_energy, show_com, save_plots, subdir_figure_eight))
        save_thread = multiprocessing.Process(target=save_anim, args=(R_figure, M, N, figure_eight_solver, subdir_figure_eight, dt, trails, show_animation))
        show_thread.start()
        save_thread.start()
        #show_thread.join()
        #save_thread.join()
    else:
        # If only one thread available, run sequentially
        make_plots(R_figure, M, N, KE_figure, PE_figure, figure_eight_solver, dt, trails, show_animation, show_energy , show_com, save_plots, subdir_figure_eight)
        save_anim(R_figure, M, N, figure_eight_solver, subdir_figure_eight, dt, trails, show_animation)

    print("Validation complete.  Results saved in Results/Validation.")
