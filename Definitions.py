from numba import float64, int32
from numba.experimental import jitclass
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
import matplotlib
import ffmpeg

G = 1. # Gravitational constant, set to 1 for convenience
# G = scipy.constants.G # physical value

class Mass:
    """
    DEPRECATED.  Used for initial testing of the n-body problem.  Not used in the final implementation.  Will be removed in future versions.
        
    """
    def __init__(self, mass, position, velocity, acceleration=np.array([0., 0.])):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
    def force(self, other_mass):
        """
        Calculate the gravitational force between two Mass objects.

        Parameters:
        other_mass (Mass): the other Mass object

        Returns:
        np.array: the gravitational force vector acting on self

        """
        # Calculate the distance between the two objects
        distance = other_mass.position - self.position

        # Calculate the gravitational force magnitude
        force_vector = (G * self.mass * other_mass.mass * distance) / np.linalg.norm(distance)**3

        return force_vector
    def update_acceleration(self, force):
        """
        Calculate and update the acceleration of the object.
        Since this is calculated exactly at every step,
        there is no need for iterative assignment.

        Parameters:
        force (np.array): the force vector acting on the object

        """
        self.acceleration = force / self.mass
    def update_velocity(self, dt):
        """
        Update the velocity of the object.

        Parameters:
        dt (double): the time step

        """
        self.velocity += self.acceleration * dt
    def update_position(self, dt):
        """
        Update the position of the object.

        Parameters:
        dt (double): the time step

        """
        self.position += self.velocity * dt
    def update(self, force, dt):
        """
        Update the position and velocity of the object.

        Parameters:
        force (np.array): the force vector acting on the object
        dt (double): the time step
        """
        self.update_acceleration(force)
        self.update_velocity(dt)
        self.update_position(dt)
    

spec = {
    'G': float64,
    'M': float64[:],
    'R0': float64[:, :],
    'V0': float64[:, :],
    't_start': float64,
    't_end': float64,
    'dt': float64,
    'N': int32,
    'T': float64[:],
    'eps': float64,
}

class OrbitSolver:
    """
    Class to solve the n-body problem.  Uses the "Leapfrog" kick-drift-kick algorithm to solve the equations of motion.

    Attributes
    ----------
    G : float
        Gravitational constant
    M : array
        Array of masses
    R0 : array
        Array of initial positions
    V0 : array
        Array of initial velocities
    t_start : float
        Start time
    t_end : float
        End time
    dt : float
        Time step
    N : int
        Number of bodies
    T : array
        Array of time steps
    eps: float
        Small number to avoid division by zero.  Default 1e-2 for numerical stability.

    Methods 
    -------
    solve()
        Solve the n-body problem using the Leapfrog algorithm. Returns the positions, velocities, kinetic energy, and potential energy of the system at each time step.
    """
    def __init__(self, G, M, R0, V0, t_start, t_end, dt, eps):
        self.G = G
        self.M = M
        self.R0 = R0
        self.V0 = V0
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.N = len(M)
        self.T = np.arange(t_start, t_end, dt)
        self.eps = eps
    def solve(self):
        """
        Solve the n-body problem using the Leapfrog algorithm
        """
        # Initialize arrays to store positions, velocities, kinetic energy, and potential energy
        R = np.zeros((self.N, 2, len(self.T)))
        V = np.zeros((self.N, 2, len(self.T)))
        KE = np.zeros(len(self.T))
        KE[0] = 0.5*np.sum(self.M*np.linalg.norm(self.V0, axis=1)**2)
        PE = np.zeros(len(self.T))
        PE[0] = self.potential_energy(self.R0)
        # Initialize the first time step
        R[:, :, 0] = self.R0
        V[:, :, 0] = self.V0
        # Loop over time steps
        for i in range(1, len(self.T)):
            # "Kick" step
            A_i = self.acceleration(R[:, :, i-1])
            V_half = V[:, :, i-1] + 0.5*self.dt*A_i
            # "Drift" step
            R[:, :, i] = R[:, :, i-1] + self.dt*V_half
            # "Kick" step
            A_i_1 = self.acceleration(R[:, :, i])
            V[:, :, i] = V_half + 0.5*self.dt*A_i_1
            # Calculate kinetic and potential energy
            KE[i] = 0.5*np.sum(self.M*np.linalg.norm(V[:, :, i], axis=1)**2)
            PE[i] = self.potential_energy(R[:, :, i])
        return R, V, KE, PE
    def acceleration(self, R):
        """
        Calculate the acceleration of the system at a given time step.
        A softening parameter ε is added to avoid division by zero.
        """
        A = np.zeros((self.N, 2))
        for i in range(self.N):
            for j in range(i):
                eff_distance_cubed = np.linalg.norm(R[j] - R[i])**3 + self.eps # speed up calculation
                A[i] += self.G*self.M[j]*(R[j] - R[i])/(eff_distance_cubed)
                A[j] += self.G*self.M[i]*(R[i] - R[j])/(eff_distance_cubed)
        return A
    def potential_energy(self, R):
        """
        Calculate the potential energy of the system at a given time step
        """
        PE = 0
        for i in range(self.N):
            for j in range(i):
                PE += -self.G*self.M[i]*self.M[j]/(np.linalg.norm(R[j] - R[i])+self.eps)
        return PE 
   
    
@jitclass(spec)
class CompiledOrbitSolver:
    
    """
    Near-identical implementation to orbit_solver, but compiled with Numba for performance.

    Attributes
    ----------
    G : float
        Gravitational constant
    M : array
        Array of masses
    R0 : array
        Array of initial positions
    V0 : array
        Array of initial velocities
    t_start : float
        Start time
    t_end : float
        End time
    dt : float
        Time step
    N : int
        Number of bodies
    T : array
        Array of time steps
    eps: float
        Small number to avoid division by zero.  Default 1e-2 for numerical stability.

    Methods 
    -------
    solve()
        Solve the n-body problem using the Leapfrog algorithm. Returns the positions, velocities, kinetic energy, and potential energy of the system at each time step.
    """
    def __init__(self, G, M, R0, V0, t_start, t_end, dt, eps):
        self.G = G
        self.M = M
        self.R0 = R0
        self.V0 = V0
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.N = len(M)
        self.T = np.arange(t_start, t_end, dt)
        self.eps = eps
    def solve(self):
        """
        Solve the n-body problem using the Leapfrog algorithm
        """
        # Initialize arrays to store positions, velocities, kinetic energy, and potential energy
        R = np.zeros((self.N, 2, len(self.T)))
        V = np.zeros((self.N, 2, len(self.T)))
        KE = np.zeros(len(self.T))
        for j in range(self.N):
            KE[0] += 0.5*self.M[j]*np.linalg.norm(self.V0[j])**2 # numba can't handle axis argument of np.linalg.norm
        PE = np.zeros(len(self.T))
        PE[0] = self.potential_energy(self.R0)
        # Initialize the first time step
        R[:, :, 0] = self.R0
        V[:, :, 0] = self.V0
        # Loop over time steps
        for i in range(1, len(self.T)):
            # "Kick" step
            A_i = self.acceleration(R[:, :, i-1])
            V_half = V[:, :, i-1] + 0.5*self.dt*A_i
            # "Drift" step
            R[:, :, i] = R[:, :, i-1] + self.dt*V_half
            # "Kick" step
            A_i_1 = self.acceleration(R[:, :, i])
            V[:, :, i] = V_half + 0.5*self.dt*A_i_1
            # Calculate kinetic and potential energy
            for j in range(self.N):
                KE[i] += 0.5*self.M[j]*np.linalg.norm(V[j, :, i])**2 # numba can't handle axis argument of np.linalg.norm
            PE[i] = self.potential_energy(R[:, :, i])
        return R, V, KE, PE
    def acceleration(self, R):
        """
        Calculate the acceleration of the system at a given time step.
        A softening parameter ε is added to avoid division by zero.
        """
        A = np.zeros((self.N, 2))
        for i in range(self.N):
            for j in range(i):
                eff_distance_cubed = np.linalg.norm(R[j] - R[i])**3 + self.eps # speed up calculation
                A[i] += self.G*self.M[j]*(R[j] - R[i])/(eff_distance_cubed)
                A[j] += self.G*self.M[i]*(R[i] - R[j])/(eff_distance_cubed)
        return A
    def potential_energy(self, R):
        """
        Calculate the potential energy of the system at a given time step
        """
        PE = 0
        for i in range(self.N):
            for j in range(i):
                PE += -self.G*self.M[i]*self.M[j]/(np.linalg.norm(R[j] - R[i])+self.eps)
        return PE 
    

"{:(|} <-- Orangutan"
   

def user_input_loop(G, N, t_start, t_end, dt, eps, seed, M_max, R_max, V_max, compiled):
    """
    Function to allow the user to change the parameters of the N-body simulation interactively.
    Exits loop when user inputs 0."""
    while True:
        # Display options to the user
        print("Choose the line number of the variable you wish to change:")
        print(f"1. G: {G}")
        print(f"2. N: {N}")
        print(f"3. t_start: {t_start}")
        print(f"4. t_end: {t_end}")
        print(f"5. dt: {dt}")
        print(f"6. eps: {eps}")
        print(f"7. Random seed: {seed}")
        print(f"8. Maximum mass: {M_max}")
        print(f"9. Maximum position: {R_max}")
        print(f"10. Maximum velocity: {V_max}")
        print(f"11. Compile the program using Numba: {compiled}")
        print("0. Run simulation with current parameters")
        print('-1. HELP')

        # Get user input
        choice = input("Enter your choice: ")

        # Check if the user wants to exit
        if choice == '0':
            print("Running program with current settings.")
            break

        # Convert choice to integer
        try:
            choice = int(choice)
        except ValueError:
            print("Invalid choice. Please enter a valid integer from 1-11 to change parameters, 0 to run the program as-is, and -1 for help.")
            continue


        # Use match statement to update the corresponding variable based on the user's choice
        match choice:
            case 1:
                new_G = input("Enter the new value for G: ") # allowed to be negative for repulsive anti-gravity.  Try it out!
                try:
                    G = float(new_G)
                except ValueError:
                    print("Invalid value. Please enter a valid float.")
                    continue
                print(f"Updated value of G: {G}")
            case 2:
                new_N = input("Enter the new value for N: ")
                try:
                    new_N = int(new_N)
                except ValueError:
                    print("Invalid value. Please enter a valid integer.")
                    continue
                if new_N <= 0:
                    print("Invalid value. Please enter a positive integer.")
                    continue
                N = new_N
                print(f"Updated value of N: {N}")
            case 3:
                new_t_start = input("Enter the new value for t_start: ")
                try:
                    new_t_start = float(new_t_start)
                except ValueError:
                    print("Invalid value. Please enter a valid float.")
                    continue
                if new_t_start < 0 or new_t_start >= t_end:
                    print("Invalid value. Please enter a non-negative float less than t_end.")
                    continue
                t_start = new_t_start
                print(f"Updated value of t_start: {t_start}")
            case 4:
                new_t_end = input("Enter the new value for t_end: ")
                try:
                    new_t_end = float(new_t_end)
                except ValueError:
                    print("Invalid value. Please enter a valid float.")
                    continue
                if new_t_end <= t_start:
                    print("Invalid value. Please enter a float greater than t_start.") # Don't need to catch 0 because it's already handled by the previous if statement
                    continue
                t_end = new_t_end
                print(f"Updated value of t_end: {t_end}")
            case 5:
                new_dt = input("Enter the new value for dt: ")
                try:
                    new_dt = float(new_dt)
                except ValueError:
                    print("Invalid value. Please enter a valid float.")
                    continue
                if new_dt <= 0:
                    print("Invalid value. Please enter a positive float.")
                    continue
                dt = new_dt
                print(f"Updated value of dt: {dt}")
            case 6:
                new_eps = input("Enter the new value for eps: ")
                try:
                    new_eps = float(new_eps)
                except ValueError:
                    print("Invalid value. Please enter a valid float.")
                    continue
                if new_eps < 0:
                    print("Invalid value. Please enter a positive float.") # negative eps can cause division by zero.  0 is allowed, but not recommended.
                    continue
                eps = new_eps
                print(f"Updated value of eps: {eps}")
            case 7:
                new_seed = (input("Enter the new value for the random seed, or 0 to regenerate the seed: "))
                try:
                    new_seed = int(new_seed)
                except ValueError:
                    print("Invalid seed, please enter a valid integer.")
                    continue
                if new_seed == 0:
                    seed = np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32)
                elif new_seed < 0:
                    print("Invalid seed. Please enter a non-negative integer.")
                    continue
                else:
                    seed = new_seed
                print(f"Updated value of seed: {seed}")
            case 8:
                new_M_max = input("Enter the new value for the maximum mass: ")
                try:
                    new_M_max = float(new_M_max)
                except ValueError:
                    print("Invalid value. Please enter a valid float.")
                    continue
                if new_M_max <= 0:
                    print("Invalid value. Please enter a positive float.")
                    continue
                M_max = new_M_max
                print(f"Updated value of maximum mass: {M_max}")
            case 9:
                new_R_max = input("Enter the new value for the maximum position: ") 
                try:
                    new_R_max = float(new_R_max)
                except ValueError:
                    print("Invalid value. Please enter a valid float.")
                    continue
                if new_R_max <= 0:
                    print("Invalid value. Please enter a positive float.")
                    continue
                R_max = new_R_max
                print(f"Updated value of maximum position: {R_max}")
            case 10:
                new_V_max = input("Enter the new value for the maximum velocity: ")
                try:
                    new_V_max = float(new_V_max)
                except ValueError:
                    print("Invalid value. Please enter a valid float.")
                    continue
                if new_V_max <= 0:
                    print("Invalid value. Please enter a positive float.")
                    continue
                V_max = new_V_max
                print(f"Updated value of maximum velocity: {V_max}")
            case 11:
                new_compiled = input("Enter 1 to compile the program using Numba, 0 to run without compilation: ")
                try:
                    new_compiled = int(new_compiled)
                except ValueError:
                    print("Invalid value. Please enter 1 or 0.")
                    continue
                if new_compiled not in [0, 1]:
                    print("Invalid value. Please enter 1 or 0.")
                    continue
                compiled = new_compiled
                print(f"Updated value of compiled: {compiled}")
            case -1:
                print("The N-body simulation simulates the motion of N bodies under the influence of gravity. The simulation contains a number of parameters that can be adjusted to change the behavior of the system. The parameters are as follows:")
                print("1. G: The gravitational constant, which determines the strength of the gravitational force between the bodies.")
                print("2. N: The number of bodies in the simulation.")
                print("3. t_start: The starting time of the simulation.")
                print("4. t_end: The ending time of the simulation.")
                print("5. dt: The time step of the simulation.")
                print("6. eps: The softening parameter, which prevents the gravitational force from becoming infinite when two bodies are close together.")
                print("7. Random seed: The seed used to generate random initial conditions for the simulation. Allows for reproducability of results.")
                print('8. Maximum mass: The maximum mass of the bodies in the simulation.  Masses will be randomly generated between 0 and this value.')
                print("9. Maximum position: The maximum initial position of the bodies in the simulation.  Positions will be randomly generated between -this value and this value.")
                print('10. Maximum velocity: The maximum velocity of the bodies in the simulation (before converting to center-of-mass frame).  Velocities will be randomly generated between -this value and this value.')
                print("11. Compile the program using Numba: If set to 1, the program will be compiled using Numba's JIT compiler for performance. If set to 0, the program will run without compilation. This is recommended for large N, but can actually slow down the program for small N-- for my machine, the tipping point was N=20, but this depends on your individual system.")
                print("To change a parameter, enter the line number of the parameter you wish to change. To run the simulation with the current parameters, enter 0.")
            case _:
                print("Invalid choice. Please enter a valid integer from 1-11 to change parameters, 0 to run the program as-is, and -1 for help.")
                continue
    return G, N, t_start, t_end, dt, eps, seed, M_max, R_max, V_max, compiled

def make_plots(R, M, N, KE, PE, solver, dt, trails, show_animation, show_energy, show_com, save_plots, subdir):
    plt.rcParams['figure.figsize'] = (10, 7)
    print("Animating...")
    t1 = time.time()
    fig1, ax1 = plt.subplots()
    ax1.set_title(rf"{N}-Body Simulation: $t_0$ = {solver.t_start}, $t_f$ = {solver.t_end}, $\Delta t$ = {solver.dt}, $\epsilon$ = {solver.eps}, $G$ = {solver.G}")
    x_min = np.min(R[:, 0, :])
    x_max = np.max(R[:, 0, :])
    ax1.set_xlim(x_min, x_max)
    y_min = np.min(R[:, 1, :])
    y_max = np.max(R[:, 1, :])
    ax1.set_ylim(y_min, y_max)
    Marker_sizes = M.copy()
    for i in range(len(M)):
        Marker_sizes[i] = max(5 * M[i] / np.max(M),0.5) # Scale the marker size based on the mass of the body, with a minimum size of 0.5 for visibility
    if trails:
        lines = [ax1.plot([], [], 'o--', markersize=Marker_sizes[i], markevery=[-1])[0] for i in range(N)] 
    else:
        lines = [ax1.plot([], [], 'o', markersize = Marker_sizes[i])[0] for i in range(N)]
    def init():
        for line in lines:
            line.set_data([], [])
        return lines
    def animate(i):
        for j, line in enumerate(lines):
            if trails:
                line.set_data([R[j, 0, max(0, i-10):i]], [R[j, 1, max(0, i-10):i]])
            else:   
                line.set_data([R[j, 0, i]], [R[j, 1, i]])
        return lines
    if show_animation: # I have no clue why this is necessary, but it is.  For some reason, if the animation is being shown, it needs to be assigned to a variable, or it won't work.  If it's not being shown, it needs to NOT be assigned to a variable.
        anim = animation.FuncAnimation(fig1, animate, frames=len(solver.T), init_func=init, blit=True, interval=dt*1000, repeat=True) # Set the interval to the time step in milliseconds
    else:
        animation.FuncAnimation(fig1, animate, frames=len(solver.T), init_func=init, blit=True, interval=dt*1000, repeat=True) # Set the interval to the time step in milliseconds
    t2 = time.time()
    print("Animation done after", t2-t1, "seconds")
    # Ensure animation is hidden if show_animation is False
    if not show_animation:
        plt.close(fig1)
    fig2, ax2 = plt.subplots()
    ax2.set_title("Energy vs. Time")
    ax2.plot(solver.T, KE, label='Kinetic Energy')
    ax2.plot(solver.T, PE, label='Potential Energy')
    ax2.plot(solver.T, KE + PE, label='Total Energy')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy')
    ax2.legend()
    if not show_energy:
        plt.close(fig2)
    if save_plots:
        print("Saving energy plot...")
        fig2.savefig(f"Results/{subdir}/energy.png")
    # Additional validation: plot the motion of the center of mass over time, should be a straight line
    COM = np.zeros((2, len(solver.T)))
    for i in range(len(solver.T)):
        COM[:, i] = np.sum(M[:, np.newaxis]*R[:, :, i], axis=0)/np.sum(M)
    fig3 = plt.figure(figsize=(20,7))
    ax31 = fig3.add_subplot(1,2,1)
    ax31.set_title("Center of Mass Motion: x vs. t")
    ax32 = fig3.add_subplot(1,2,2)
    ax32.set_title("Center of Mass Motion: y vs. t")
    ax31.plot(solver.T, COM[0])
    ax31.set_xlabel('Time')
    ax31.set_ylabel('x')
    ax31.set_title('Center of Mass x Motion')
    ax31.set_ylim(-1, 1)
    ax32.plot(solver.T, COM[1])
    ax32.set_xlabel('Time')
    ax32.set_ylabel('y')
    ax32.set_title('Center of Mass y Motion')
    ax32.set_ylim(-1, 1)
    fig3.tight_layout()
    if not show_com:
        plt.close(fig3)
    if save_plots:
        print("Saving center of mass plot...")
        fig3.savefig(f"Results/{subdir}/center_of_mass.png")
    plt.show()
    print("Plots closed.  Terminating process.")

def save_anim(R, M, N, solver, subdir, dt, trails, show_animation):
    figx, axx = plt.subplots()
    axx.set_title(rf"{N}-Body Simulation: $t_0$ = {solver.t_start}, $t_f$ = {solver.t_end}, $\Delta t$ = {solver.dt}, $\epsilon$ = {solver.eps}, $G$ = {solver.G}")
    x_min = np.min(R[:, 0, :])
    x_max = np.max(R[:, 0, :])
    axx.set_xlim(x_min, x_max)
    y_min = np.min(R[:, 1, :])
    y_max = np.max(R[:, 1, :])
    axx.set_ylim(y_min, y_max)
    Marker_sizes = M.copy()
    for i in range(len(M)):
        Marker_sizes[i] = max(5 * M[i] / np.max(M),0.5) # Scale the marker size based on the mass of the body, with a minimum size of 0.5 for visibility
    if trails:
        lines = [axx.plot([], [], 'o--', markersize=Marker_sizes[i], markevery=[-1])[0] for i in range(N)] 
    else:
        lines = [axx.plot([], [], 'o', markersize = Marker_sizes[i])[0] for i in range(N)]
    def init():
        for line in lines:
            line.set_data([], [])
        return lines
    def animate(i):
        for j, line in enumerate(lines):
            if trails:
                line.set_data([R[j, 0, max(0, i-10):i]], [R[j, 1, max(0, i-10):i]])
            else:
                line.set_data([R[j, 0, i]], [R[j, 1, i]])
        return lines
    ani = animation.FuncAnimation(figx, animate, frames=len(solver.T), init_func=init, blit=True, interval=dt*1000, repeat=True) # Set the interval to the time step in milliseconds
    # Save the animation to a file
    #plt.close(figx)
    t2 = time.time()
    print("Saving animation...")
    ani.save(f"Results/{subdir}/animation.mp4", writer='ffmpeg',fps=1/dt, dpi=200)
    t3 = time.time()
    print("Animation saved after", t3-t2, "seconds")

NBody = """ 
.-----------------. .----------------.  .----------------.  .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
| | ____  _____  | || |              | || |   ______     | || |     ____     | || |  ________    | || |  ____  ____  | |
| ||_   \|_   _| | || |              | || |  |_   _ \    | || |   .'    `.   | || | |_   ___ `.  | || | |_  _||_  _| | |
| |  |   \ | |   | || |    ______    | || |    | |_) |   | || |  /  .--.  \  | || |   | |   `. \ | || |   \ \  / /   | |
| |  | |\ \| |   | || |   |______|   | || |    |  __'.   | || |  | |    | |  | || |   | |    | | | || |    \ \/ /    | |
| | _| |_\   |_  | || |              | || |   _| |__) |  | || |  \  `--'  /  | || |  _| |___.' / | || |    _|  |_    | |
| ||_____|\____| | || |              | || |  |_______/   | || |   `.____.'   | || | |________.'  | || |   |______|   | |
| |              | || |              | || |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------' """
