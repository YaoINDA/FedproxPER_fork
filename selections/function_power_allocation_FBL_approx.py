#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cvxpy as cp

def error_prob_fbl_approx(snr, blocklength, packet_size):
    """
    Calculate approximation of error probability for finite blocklength communications.
    
    Parameters:
    -----------
    snr : float or numpy.ndarray
        Signal-to-noise ratio (gamma)
    blocklength : float or numpy.ndarray
        Blocklength (m)
    packet_size : float or numpy.ndarray
        Packet size (D)
    
    Returns:
    --------
    err_approx : float or numpy.ndarray
        Approximated error probability
    """
    # Calculate alpha parameter
    alpha = np.exp(packet_size / blocklength) - 1
    
    # Calculate mu parameter
    mu = np.sqrt(blocklength / (np.exp(2 * packet_size / blocklength) - 1))
    
    # Calculate error approximation
    err_approx = 0.5 - (mu / np.sqrt(2 * np.pi)) * (snr - alpha)
    
    # Ensure results are in valid range [0,1]
    if isinstance(err_approx, np.ndarray):
        err_approx = np.clip(err_approx, 0, 1)
    else:
        err_approx = max(0, min(1, err_approx))
    
    return err_approx

def power_allocation_fbl_approx(h_i, weights, later_weights, data_size, blocklength, 
                                N0, B, P_max, P_sum, theta=0, Tslot=1.0):
    """
    Solve power allocation problem using CVXPY with linear approximation of error probability.
    
    Parameters:
    -----------
    h_i : numpy.ndarray
        Channel gains for each user
    weights : numpy.ndarray
        Weight for each user in the objective function
    later_weights : numpy.ndarray
        Additional weights for each user
    data_size : numpy.ndarray
        Data size for each user in bits
    blocklength : numpy.ndarray or float
        Blocklength allocated to each user or single value for all users
    N0 : float
        Noise power spectral density
    B : float
        Bandwidth
    P_max : float
        Maximum power per user
    P_sum : float
        Sum power constraint
    theta : float, optional
        Energy coefficient related to computation
    Tslot : float, optional
        Time slot duration
        
    Returns:
    --------
    P_opt : numpy.ndarray
        Optimal power allocation
    obj_value : float
        Objective function value at the optimal solution
    status : str
        Status of the optimization problem
    """
    K = len(h_i)
    
    # Ensure blocklength is an array
    if np.isscalar(blocklength):
        blocklength = np.ones(K) * blocklength
    
    # Define power variable
    P = cp.Variable(K)
    
    # Calculate SNR denominator for each user
    D_i = N0 * B / h_i
    
    # Pre-calculate approximation parameters
    alpha_values = np.zeros(K)
    mu_values = np.zeros(K)
    
    for i in range(K):
        alpha_values[i] = np.exp(data_size[i] / blocklength[i]) - 1
        mu_values[i] = np.sqrt(blocklength[i] / (np.exp(2 * data_size[i] / blocklength[i]) - 1))
    
    # Define constraints
    constraints = [
        P >= 0,                  # Power must be non-negative
        P <= P_max,              # Power per user must not exceed maximum
        cp.sum(P) + theta/Tslot * cp.sum(data_size) <= P_sum,  # Total power constraint
    ]
    
    # Add constraint to ensure error probability is non-negative
    for i in range(K):
        # Lower bound constraint (error probability >= 0)
        constraints.append(0.5 - (mu_values[i] / np.sqrt(2 * np.pi)) * (P[i] / D_i[i] - alpha_values[i]) >= 0)
    
    # Define the error probability expression for each user
    err_prob_expressions = []
    for i in range(K):
        # Calculate error probability expression
        err_prob_i = 0.5 - (mu_values[i] / np.sqrt(2 * np.pi)) * (P[i] / D_i[i] - alpha_values[i])
        err_prob_expressions.append(err_prob_i)
    
    # Convert the list of expressions to a vector expression
    err_prob_vector = cp.vstack(err_prob_expressions)
    
    # Objective function with error probability as a function of P
    objective_expression = cp.sum(cp.multiply(weights, err_prob_vector)) - cp.sum(later_weights)
    objective = cp.Minimize(objective_expression)
    
    # Create the problem
    prob = cp.Problem(objective, constraints)
    
    # Try different solvers in order of preference
    solvers = [cp.MOSEK, cp.ECOS, cp.SCS, None]  # None lets CVXPY choose
    solution_found = False
    
    for solver in solvers:
        try:
            if solver is None:
                prob.solve()
            else:
                prob.solve(solver=solver)
            
            solution_found = prob.status in ["optimal", "optimal_inaccurate"]
            if solution_found:
                break
        except:
            continue
    
    # Check if the problem was solved successfully
    if not solution_found:
        return None, float('inf'), prob.status
    
    return P.value, prob.value, prob.status