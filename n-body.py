import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import ffmpeg # Writer for saving animations to .mp4 file.  Pylance and the like may think this variable is unused, but it is necessary for saving animations.
import os
import multiprocessing
from Definitions import *


if __name__ == '__main__': # Must be in the main block to allow for multiprocessing
    # Report number of available threads
    num_threads = os.cpu_count()
    print("Number of available threads:", num_threads),
    if num_threads == 1:
        print("Will not be able to use multithreading to speed up animation generation.  Animation will not generate until figures have been closed.")
    else:
        print("Multithreading available.  Animation will generate in the background.")

    if not os.path.exists("Results"):
        os.makedirs("Results")


    # Change global figsize to 10x7
    plt.rcParams['figure.figsize'] = (10, 7)
    print(NBody)
    print("Welcome to the N-Body Simulation! Originally written by Jack Felice in fulfillment of the final project requirement for OSU Physics 5810: Computational Physics.")


    # Constants
    G = 1.0 # Gravitational constant
    N = 10 # Number of bodies
    # Time 
    t_start = 0.0
    t_end = 10.0
    dt = 0.01
    seed = 42

    # Softening parameter
    eps = 1e-2

    # Maximum components for random mass, position, and velocity generation
    M_max = 1.0
    R_max = 1.0
    V_max = 1.0

    # Variable to track whether the user wants to compile the program using Numba's JIT compiler
    compiled = 0

    # Take user input for all parameters
    G, N, t_start, t_end, dt, eps, seed, M_max, R_max, V_max, compiled = user_input_loop(G, N, t_start, t_end, dt, eps, seed, M_max, R_max, V_max, compiled)

    # Seed the random number generator for reproducibility
    np.random.seed(seed)
    M = np.random.uniform(0, M_max, N) # Random masses
    R = np.random.uniform(-R_max, R_max, (N, 2)) # Random positions in 2D
    V =  np.random.uniform(-V_max, V_max, (N, 2)) # Random velocities in 2D

    # Convert to center-of-mass frame to prevent drift
    R_com = np.sum(M[:, np.newaxis]*R, axis=0)/np.sum(M)
    V_com = np.sum(M[:, np.newaxis]*V, axis=0)/np.sum(M)
    V -= V_com
    R -= R_com
    #Check that the center of mass is at rest (within numerical error)
    print("Center of mass velocity:", np.sum(M[:, np.newaxis]*V, axis=0)/np.sum(M)) 


    t0 = time.time()
    # Initialize the solver
    class_args = (G, M, R, V, t_start, t_end, dt, eps)
    if compiled:
        solver = CompiledOrbitSolver(*class_args)
    else:  
        solver = OrbitSolver(*class_args)
    print("Solving...")
    R, V, KE, PE = solver.solve()
    t1 = time.time()
    print("Motion solved after", t1-t0, "seconds")

    print("Which plots would you like to see? Set 0 for no, 1 for yes.")
    show_animation = 1
    show_energy = 1
    show_com = 1
    save_plots = 0
    while True:
        print(f"1. Animation: {show_animation}")
        print(f"2. Energy: {show_energy}")
        print(f"3. Center of Mass: {show_com}")
        print(f"4. Save plots: {save_plots}")
        print("0. Continue")
        choice = input("Enter your choice: ")
        try:
            choice = int(choice)
        except ValueError:
            print("Invalid choice. Please enter a valid integer from 0-3.")
            continue
        if choice == 0:
            break
        match choice:
            case 1:
                show_animation = int(input("Enter 1 to show the animation, 0 to hide it: "))
            case 2:
                show_energy = int(input("Enter 1 to show the energy plot, 0 to hide it: "))
            case 3:
                show_com = int(input("Enter 1 to show the center of mass plot, 0 to hide it: "))
            case 4:
                save_plots = int(input("Enter 1 to save the plots, 0 to not save them: "))
            case _:
                print("Invalid choice. Please enter 1-3 to change parameters, or 0 to continue.")

    if save_plots:
        subdir = f"{N}-body_G={G},t_start={t_start},t_end={t_end},dt={dt},eps={eps},M_max={M_max},R_max={R_max},V_max={V_max},seed={seed}"
        if os.path.exists(f"Results/{subdir}"):
            print("Directory already exists. Overwriting...")
        os.makedirs(f"Results/{subdir}", exist_ok=True)




    while True:
        trails = input("Would you like to see trails of the bodies? Set 0 for no, 1 for yes.")
        try:
            trails = int(trails)
        except ValueError:
            print("Invalid choice. Please enter 0 or 1.")
            continue
        if trails == 0 or trails == 1:
            break
        else:
            print("Invalid choice. Please enter 0 or 1.")


    if save_plots: # Allow the animation to be saved in parallel with the plots being shown, as the animation rendering takes the longest time (especially if compiled!)
        if num_threads >= 2:
            # Create and start the threads
            show_thread = multiprocessing.Process(target=make_plots, args=(R, M, N, KE, PE, solver, dt, trails, show_animation, show_energy, show_com, save_plots, subdir))
            save_thread = multiprocessing.Process(target=save_anim, args=(R, M, N, solver, subdir, dt, trails, show_animation))
            show_thread.start()
            save_thread.start()
            show_thread.join()
            save_thread.join()
        else:
            # If only one thread available, run sequentially
            make_plots(R, M, N, KE, PE, solver, dt, trails, show_animation, show_energy , show_com, save_plots, subdir)
            save_anim(R, M, N, solver, subdir, dt, trails, show_animation)
    else:
        subdir = "temp" # should never be created, assuming not(True) is False
        make_plots(R, M, N, KE, PE, solver, dt, trails, show_animation, show_energy, show_com, save_plots, subdir)