import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import ffmpeg

from Definitions import OrbitSolver

# Change global figsize to 10x7
plt.rcParams['figure.figsize'] = (10, 7)
print("""
 .----------------.  .----------------.  .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
| |    ______    | || |   ______     | || |     ____     | || |  ________    | || |  ____  ____  | |
| |   / ____ `.  | || |  |_   _ \    | || |   .'    `.   | || | |_   ___ `.  | || | |_  _||_  _| | |
| |   `'  __) |  | || |    | |_) |   | || |  /  .--.  \  | || |   | |   `. \ | || |   \ \  / /   | |
| |   _  |__ '.  | || |    |  __'.   | || |  | |    | |  | || |   | |    | | | || |    \ \/ /    | |
| |  | \____) |  | || |   _| |__) |  | || |  \  `--'  /  | || |  _| |___.' / | || |    _|  |_    | |
| |   \______.'  | || |  |_______/   | || |   `.____.'   | || | |________.'  | || |   |______|   | |
| |              | || |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------'  '----------------' """)
print("Welcome to the N-Body Simulation!")


# Constants
G = 1 # Gravitational constant
N = 10 # Number of bodies
# Time 
t_start = 0
t_end = 10
dt = 0.01

# Softening parameter
eps = 1e-2

while True:
    # Display options to the user
    print("Choose a variable to change:")
    print(f"1. G: {G}")
    print(f"2. N: {N}")
    print(f"3. t_start: {t_start}")
    print(f"4. t_end: {t_end}")
    print(f"5. dt: {dt}")
    print("0. Run simulation with current parameters")

    # Get user input
    choice = input("Enter your choice: ")

    # Check if the user wants to exit
    if choice == '0':
        print("Exiting program.")
        break

    # Convert choice to integer
    choice = int(choice)


    # Use match statement to update the corresponding variable based on the user's choice
    match choice:
        case 1:
            G = float(input("Enter the new value for G: "))
            print(f"Updated value of G: {G}")
        case 2:
            N = int(input("Enter the new value for N: "))
            print(f"Updated value of N: {N}")
        case 3:
            t_start = float(input("Enter the new value for t_start: "))
            print(f"Updated value of t_start: {t_start}")
        case 4:
            t_end = float(input("Enter the new value for t_end: "))
            print(f"Updated value of t_end: {t_end}")
        case 5:
            dt = float(input("Enter the new value for dt: "))
            print(f"Updated value of dt: {dt}")
        case _:
            print("Invalid choice. Please enter 1-5 to change parameters, or 0 to exit.")
M = np.random.rand(N) # Masses
R = np.random.rand(N, 2) # Initial positions in 2D
V = np.random.rand(N, 2) # Initial velocities in 2D
# Convert to center-of-mass frame to prevent drift
R_com = np.sum(M[:, np.newaxis]*R, axis=0)/np.sum(M)
V_com = np.sum(M[:, np.newaxis]*V, axis=0)/np.sum(M)
V -= V_com
R -= R_com
#Check that the center of mass is at rest
print("Center of mass velocity:", np.sum(M[:, np.newaxis]*V, axis=0)/np.sum(M)) 

# Debugging: set initial positions and velocities to known values, vertices of equilateral triangle with velocities tangent to circle
# R = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
# V = 0.05 * np.array([[-1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), -1/np.sqrt(2)], [1, 0]])
# 


t0 = time.time()
# Initialize the solver
solver = OrbitSolver(G, M, R, V, t_start, t_end, dt, eps)
print("Solving...")
R, V, KE, PE = solver.solve()
t1 = time.time()
print("Motion solved after", t1-t0, "seconds")

# Plot the positions of the bodies
print("Animating...")
fig, ax = plt.subplots()
x_min = np.min(R[:, 0, :])
x_max = np.max(R[:, 0, :])
ax.set_xlim(x_min, x_max)
y_min = np.min(R[:, 1, :])
y_max = np.max(R[:, 1, :])
ax.set_ylim(y_min, y_max)
lines = [ax.plot([], [], 'o')[0] for i in range(N)]
def init():
    for line in lines:
        line.set_data([], [])
    return lines
def animate(i):
    for j, line in enumerate(lines):
        line.set_data(R[j, 0, i], R[j, 1, i])
    return lines
ani = animation.FuncAnimation(fig, animate, frames=len(solver.T), init_func=init, blit=True, interval=dt*1000, repeat=True) # Set the interval to the time step in milliseconds
t2 = time.time()
print("Animation done after", t2-t1, "seconds")
# Save the animation to a file
print("Saving animation...")
#ani.save('n-body.mp4', writer='ffmpeg', fps=1/dt)
t3 = time.time()
print("Animation saved after", t3-t2, "seconds")
plt.show()
#Debug: plot the inital positions and acceleration vectors
# fig, ax = plt.subplots()
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# print("Initial positions:", R[:, :, 0])
# print("Initial velocities:", V[:, :, 0])
# print("Initial accelerations:", solver.acceleration(R[:, :, 0]))
# for i in range(N):
#     ax.plot(R[i, 0, 0], R[i, 1, 0], 'o')
#     ax.arrow(R[i, 0, 0], R[i, 1, 0], solver.acceleration(R[:, :, 0])[i, 0]/10, solver.acceleration(R[:, :, 0])[i, 1]/10)
# plt.show()
# Plot the kinetic and potential energy of the system over time
fig, ax = plt.subplots()
ax.plot(solver.T, KE, label='Kinetic Energy')
ax.plot(solver.T, PE, label='Potential Energy')
ax.plot(solver.T, KE + PE, label='Total Energy')
ax.set_xlabel('Time')
ax.set_ylabel('Energy')
ax.legend()
plt.show()
# Additional validation: plot the motion of the center of mass over time, should be a straight line
COM = np.zeros((2, len(solver.T)))
for i in range(len(solver.T)):
    COM[:, i] = np.sum(M[:, np.newaxis]*R[:, :, i], axis=0)/np.sum(M)
fig = plt.figure(figsize=(20,7))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.plot(solver.T, COM[0])
ax1.set_xlabel('Time')
ax1.set_ylabel('x')
ax1.set_title('Center of Mass x Motion')
ax1.set_ylim(-1, 1)
ax2.plot(solver.T, COM[1])
ax2.set_xlabel('Time')
ax2.set_ylabel('y')
ax2.set_title('Center of Mass y Motion')
ax2.set_ylim(-1, 1)
fig.tight_layout()
plt.show()
