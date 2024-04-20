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
- [ ] Check solutions of motion against special cases of the 3 body problem
- [X] Investigate scaleability improvements (e.g. NJIT, Cython)
- [X] Convert to center-of-mass frame so that motion is visible at all times
- [X] Check conservation of energy as an additional form of validation
- [ ] (Bonus, if time permits) See if a neural network can predict stability of configurations
