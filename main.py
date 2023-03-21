#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import src.ssm as ssm
from src.functions import *

def f_solid(w, vec, **kwargs):
    # parameters
    parameters = kwargs.get('parameters',[None,None]) 
    
    parameters_layers = parameters[0]
    parameters_solidBG = parameters[1]

    # 
    X = calc_X(parameters_layers.N, w, parameters_layers)
    Y = calc_Y_solid(parameters_layers.N, w, parameters_solidBG, parameters_layers.Rs[-1])
    Z = calc_Z_solid(parameters_layers.N, w, parameters_layers)
    amat = np.block([Z@X, -Y])

    return np.linalg.inv(amat) @ vec

def f_fluid(w, vec, **kwargs):
    # parameters
    parameters = kwargs.get('parameters',[None,None])
    
    parameters_layers = parameters[0]
    parameters_fluidBG = parameters[1]

    # 
    X = calc_X(parameters_layers.N, w, parameters_layers)
    Y = calc_Y_fluid(parameters_layers.N, w, parameters_fluidBG, parameters_layers.Rs[-1])
    Z = calc_Z_fluid(parameters_layers.N, w, parameters_layers)
    amat = np.block([Z@X, -Y])

    return np.linalg.inv(amat) @ vec

if __name__ == "__main__":

    #######################################
    # Set material and geometry parameters
    #######################################
    parameters_layers = Parameters_layers(
        8, # azimuthal index
        3, # Number of layers (including background)
        [0.5,1.0], # Radii (from innermost to outermost)
        [16.0,4.0], # mass density (from innermost to outermost)
        [1.0,1.0], # Lame's constants mu
        [0.3,0.3], # Poisson's ratio
    )

    parameters_solidBG = Parameters_solidBG(
        1.0, # exterior mass density
        1.0, # exterior Lame's constant mu
        0.3, # exterior Poisson's ratio
    )

    parameters_fluidBG = Parameters_fluidBG(
        1.0, # exterior mass density
        1.0, # exterior bulk modulus
    )

    #############################################
    # Scattering poles in the solid-solid system
    #############################################
    s = ssm.SSM(2, 8, 1e-5, 1e-5, 4)
    # search scattering poles w within the circle centered at 5.25-0.1j with radius 0.4 on the complex-w plane.
    eigenvalues, eigenvectors = s.run(f_solid, 800, 0.4, 5.25-0.1j, parameters=[parameters_layers,parameters_solidBG])

    for eigenvalue,eigenvector in zip(eigenvalues,eigenvectors):
        print("")
        print("# Eigenvalue:")
        print(eigenvalue)
        print("# Eigenvector: ")
        print(eigenvector)
        print("")


    #############################################
    # Scattering poles in the solid-fluid system
    #############################################
    s = ssm.SSM(2, 8, 1e-5, 1e-5, 3)
    # search scattering poles w within the circle centered at 6.7-0.1j with radius 0.6 on the complex-w plane.
    eigenvalues, eigenvectors = s.run(f_fluid, 800, 0.6, 6.7-0.1j, parameters=[parameters_layers,parameters_fluidBG])

    for eigenvalue,eigenvector in zip(eigenvalues,eigenvectors):
        print("")
        print("# Eigenvalue:")
        print(eigenvalue)
        print("# Eigenvector: ")
        print(eigenvector)
        print("")



    
                
                



    

