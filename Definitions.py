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
    
    
    