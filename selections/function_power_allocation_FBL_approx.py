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

def power_allocation_fbl_approx(h_i, weights, later_weights, packet_size, blocklength, 
                               N0, B, P_max, P_sum, theta=0, Tslot=1.0):
    """
    Solve power allocation problem using CVXPY with linear approximation of error probability.
    """
    print("\n=== POWER ALLOCATION FBL APPROX DIAGNOSTICS ===")
    K = len(h_i)
    print(f"Number of users: {K}")
    print(f"h_i: {h_i}")
    print(f"weights: {weights}")
    print(f"later_weights: {later_weights}")
    print(f"packet_size: {packet_size}")
    print(f"blocklength: {blocklength}")
    print(f"N0: {N0}")
    print(f"B: {B}")
    print(f"P_max: {P_max}")
    print(f"P_sum: {P_sum}")
    print(f"theta: {theta}")
    print(f"Tslot: {Tslot}")
    
    # Ensure blocklength is an array
    if np.isscalar(blocklength):
        blocklength = np.ones(K) * blocklength
    
    # Define power variable
    P = cp.Variable(K)
    
    # Calculate SNR denominator for each user
    D_i = N0 * B / h_i
    print(f"D_i: {D_i}")
    
    # Pre-calculate approximation parameters
    alpha_values = np.zeros(K)
    mu_values = np.zeros(K)
    
    for i in range(K):
        try:
            alpha_values[i] = np.exp(packet_size[i] / blocklength[i]) - 1
            mu_values[i] = np.sqrt(blocklength[i] / (np.exp(2 * packet_size[i] / blocklength[i]) - 1))
        except Exception as e:
            print(f"Error calculating parameters for user {i}: {e}")
            raise
    
    print(f"alpha_values: {alpha_values}")
    print(f"mu_values: {mu_values}")
    
    # Define constraints
    constraints = [
        P >= 0,                  # Power must be non-negative
        P <= P_max,              # Power per user must not exceed maximum
        #cp.sum(P) + theta/Tslot * cp.sum(packet_size) <= P_sum,  # Total power constraint
        cp.sum(P) <= P_sum,
    ]
    
    print(f"Power constraint: sum(P) <= {P_sum}")
    print(f"Computation power term: {theta/Tslot * np.sum(packet_size)}")
    
    # Add constraint to ensure error probability is non-negative
    for i in range(K):
        try:
            # Lower bound constraint (error probability >= 0)
            epsilon = 1e-6
            constraints.append(0.5 - (mu_values[i] / np.sqrt(2 * np.pi)) * (P[i] / D_i[i] - alpha_values[i]) >=  -epsilon)
        except Exception as e:
            print(f"Error adding constraint for user {i}: {e}")
            raise
    
    # Define the error probability expression for each user
    err_prob_expressions = []
    for i in range(K):
        try:
            # Calculate error probability expression
            err_prob_i = 0.5 - (mu_values[i] / np.sqrt(2 * np.pi)) * (P[i] / D_i[i] - alpha_values[i])
            err_prob_expressions.append(err_prob_i)
        except Exception as e:
            print(f"Error defining error probability for user {i}: {e}")
            raise
    
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
            print(f"Trying solver: {solver}")
            if solver is None:
                prob.solve(verbose=True)
            else:
                prob.solve(solver=solver, verbose=True)
            
            solution_found = prob.status in ["optimal", "optimal_inaccurate"]
            print(f"Solver result: {prob.status}")
            if solution_found:
                break
        except Exception as e:
            print(f"Solver {solver} failed with error: {e}")
            continue
    
    # Check if the problem was solved successfully
    if not solution_found:
        print(f"All solvers failed. Final status: {prob.status}")
        return None, float('inf'), prob.status
    
    print(f"Optimal power allocation: {P.value}")
    return P.value, prob.value, prob.status