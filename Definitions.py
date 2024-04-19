import numpy as np
import scipy

G = 1. # Gravitational constant, set to 1 for convenience
# G = scipy.constants.G # physical value

class Mass:
    """
    Mass object to use for N-Body Simulation.

    Attributes:
    mass (double): mass of the object
    position (np.array): position of the object (in 2D)
    velocity (np.array): velocity of the object (in 2D)

        
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
    ### TODO: Implement using differential equation solver, such as scipy.integrate
    ### TODO: Implement Kinetic and Potential Energy calculations to check conservation of energy
    
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
        Calculate the acceleration of the system at a given time step
        """
        A = np.zeros((self.N, 2))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    A[i] += self.G*self.M[j]*(R[j] - R[i])/(np.linalg.norm(R[j] - R[i])**3+self.eps)
        return A
    def potential_energy(self, R):
        """
        Calculate the potential energy of the system at a given time step
        """
        PE = 0
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    PE += -self.G*self.M[i]*self.M[j]/(np.linalg.norm(R[j] - R[i])+self.eps)
        return PE/2 # Divide by 2 to avoid double counting of each interaction
   
    
    