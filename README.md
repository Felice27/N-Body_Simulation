# Jack Felice 5810 Final Project: N-Body Simulation

The contents of this repository are intended to satisfy the requirements for the "final project" in OSU Physics 5810: Computational Physics.  This project was inspired by Cixin Liu's 三体 (commonly known as the *Three Body Problem* in English).

## Project Overview

The purpose of this project is to implement a physics-based code for the generalized N-body problem: N masses under mutual gravitational attraction.  In general, the motion of each body is governed by:

$$ 
\ddot{\mathbf{r}}\_i = \sum_{j=1,i \neq j}^N \frac{G m_j \left(\mathbf{r}_j - \mathbf{r}_i \right)}{|| \mathbf{r}_j - \mathbf{r}_i||^3}
$$

(Note the distinction between the vector $\mathbf{r}$ and the scalar $r$).  For the case of the 3-body problem, this gives 3 second-order differential equations:

$$
\ddot{\mathbf{r}}_1 = \frac{G m_2 \left(\mathbf{r}_2 - \mathbf{r}_1 \right)}{|| \mathbf{r}_2 - \mathbf{r}_1||^3} + \frac{G m_3 \left(\mathbf{r}_3 - \mathbf{r}_1 \right)}{|| \mathbf{r}_3 - \mathbf{r}_1||^3}
$$

$$
\ddot{\mathbf{r}}_2 = \frac{G m_1 \left(\mathbf{r}_1 - \mathbf{r}_2 \right)}{|| \mathbf{r}_1 - \mathbf{r}_2||^3} + \frac{G m_3 \left(\mathbf{r}_3 - \mathbf{r}_2 \right)}{|| \mathbf{r}_3 - \mathbf{r}_2||^3}
$$

$$
\ddot{\mathbf{r}}_3 = \frac{G m_1 \left(\mathbf{r}_1 - \mathbf{r}_3 \right)}{|| \mathbf{r}_1 - \mathbf{r}_3||^3} + \frac{G m_2 \left(\mathbf{r}_2 - \mathbf{r}_3 \right)}{|| \mathbf{r}_2 - \mathbf{r}_3||^3}
$$

Using these differential equations, along with each mass's initial position and momentum, we can solve for the motion of the particle over time and examine the behavior of the system.  I intend to work in units where $G=1$, as this will make pronounced gravitational effects appear at much more reasonable values of each mass and position. Additionally, I plan to restrict our coordinate system to $\mathbb{R}^2$, as this allows all solutions of motion to be easily visualized and plotted (which is also in-line with the physical phenomenon of conservation of angular momentum restricting solar systems and galaxies to coplanar disks), but I may experiment with 3-dimensional solar systems as well.  

## Setup

To run this program, first clone the repository.  I used Conda to manage my packages and have provided the YAML files for both Windows and Linux.  First, install Conda (either through Anaconda, Miniconda, whatever).  Then:

For Windows:
Open an Anaconda prompt, and type the following command:
```
conda env create -f NBody.yml
```
Following the subsquent installation process will create an environment called "NBody" that has all the required packages.  Then, activate the environment:
```
conda activate NBody
```

For Linux:
Open a terminal, then type the following command:
```
conda env create -f NBody-Linux.yml
```
Following the subsquent installation process will create an environment called "NBody" that has all the required packages.  Then, activate the environment:
```
conda activate NBody-Linux
```

For Mac OS X:
I actually don't know how this would work, and I have no Mac with which to experiment.  However, I know that OS X is a Unix OS, so I would wager creating the environment would go something like:
```
conda create -n NBody python=3.11 numpy scipy matplotlib jupyter ipympl ipywidgets numba ffmpeg-python
```
At which point, it's just a simple 
```
conda activate NBody
```
Away.

Now, regardless of OS, you should have a conda environment with all the required packages.

## Instructions

In your terminal / PowerShell / cmd prompt, make sure you are in your local copy of the N-Body_Simulation directory with your NBody conda environment active.  This project consists of 2 primary programs:

1. n-body.py, a command-line Python program for performing N-Body Simulation.
2. Interactive.ipynb, a Jupyter Notebook for adjusting simulation parameters and viewing animations in real time.

To run the former:
```
python n-body.py
```
And to run the latter, start a JupyterLab session:
```
jupyter lab
```
(You could also start a Jupyter Notebook session if you prefer).  Then, use the Jupyter UI to navigate to and run Interactive.ipynb.

## Goals
This project is intended to be completed in Python with interactive Jupyter notebooks, possibly using njit or Cython for computational improvements.  This project has a few long-term goals and intermediate objectives to accomplish:

- [x] Make Github Repository
- [x] Make classes for mass objects
- [X] Investigate ways to parallelize calculations (e.g. matrix operations)
- [X] Make YAML / requirements.txt file for necessary Python environment to run code locally
- [x] Implement animations for solutions of motion
- [x] Check solutions of motion against special cases of the 3 body problem
- [X] Investigate scaleability improvements (e.g. NJIT, Cython)
- [X] Convert to center-of-mass frame so that motion is visible at all times
- [X] Check conservation of energy as an additional form of validation
- [ ] (Bonus, if time permits) See if a neural network can predict stability of configurations

## Notes on changes made for v1.1.0
In response to feedback on my initial attempt, I have made a few changes to the program.  

### Properly Implementing Softening
The first was to properly implement the gravitational potential to be conservative: that is, for any interaction between two masses i and j, the contribution to potential energy is given by:

$$
U = -\frac{G m_i m_j}{\sqrt{r_{ij}^2 + \epsilon^2}}, \text{where } r_{ij} = ||\mathbf{r}_{ij}|| = ||\mathbf{r}\_j - \mathbf{r}\_i||
$$

As the force is the negative gradient of the potential, this means that the force on mass $i$ due to mass $j$ is given as follows:

$$
\mathbf{F}\_{ij} = \frac{G m_i m_j \cdot \mathbf{r}\_{ij}}{\left({r\_{ij}^2 + \epsilon^2}\right)^{3/2}}
$$

This vector field is properly conservative and, given the right choices for $\epsilon$ and the time step $\Delta t$, will exhibit conservation of energy.  An example of such a plot generated during a run of "validation.py" is shown:

![A plot with two subplots, one showing the kinetic and potential energy oscillating to conserve total energy, and one showing the relative error in energy vs time as an n-body simulation runs.  The relative error increases up to a maximum error, where it oscillates under that bound potential indefinitely.](/Figures/energy.png?raw=true "Energy Error Plot")

The plot on the left shows the kinetic, potential, and total energy evolving over time, with the variations in kinetic and potential energy roughly cancelling out to keep the total energy constant.  On the plot of relative error in total energy vs time on the right, we can see that, as advertised, the Leapfrog solver is symplectic: even though the relative error in energy initially increases, it oscillates under a maximum envelope, never exceeding roughly $10^{-4}$ in magnitude.  

### Validation: Known Behavior

The file "validation.py" was created to create some plots showing the program simulate some systems with known behavior.  The file took about 3 minutes for my PC to run, with most of that time spent rendering the animations, but this varies greatly on a machine-to-machine basis; I may have accidentally run it on a full ASCEND node, which rendered the animations in less than a second.  3 validation cases are tested:

Two masses orbiting their center of mass:

![An animation displaying two identical masses orbiting their center of mass in a perfectly circular orbit.](/Figures/two_body.gif?raw=true "Two Masses in Circular Orbit")

Three masses in an "Euler's triangle," forming the endpoints of a rotating equilateral triangle that circumscribes a circle:

![An animation displaying three identical masses orbiting their center of mass in a perfectly circular orbit.](/Figures/euler.gif?raw=true "Three Masses in Circular Orbit")

And finally, the most interesting case (at least to me): a stable, periodic solution first discovered in [Moore (1993)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.70.3675) and later proven formally in [Chenciner and Montgomery (2000)](https://arxiv.org/abs/math/0011268).  3 identical masses start in a colinear configuration, but this is not the trivial colinear case first discovered by Euler (which is periodic but unstable); instead, the masses trace out a figure-eight in a stable solution, meaning that small perturbations return to this island of stability.

![An animation displaying three identical masses in a stable figure-eight orbit.](/Figures/figure_eight.gif?raw=true "Three Masses in a Figure-Eight Orbit")

With conservation of energy and these known configurations successfully demonstrated, the validity of the model has been affirmed.  I hope you enjoy using this program-- if, by some chance, you are some observer years in the future who stumbled across this repository, feel free to shoot me a message and let me know that you came across this project.  Happy simulating!