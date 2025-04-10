#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse

def dB2power(x):
    """Convert dB to linear power"""
    return np.exp(x / 10 * np.log(10))

def error_prob_fbl(snr, blocklength, rate):
    """
    Calculate the error probability for finite blocklength communications.
    
    Parameters:
    -----------
    snr : float or numpy.ndarray
        Signal-to-noise ratio
    blocklength : float or numpy.ndarray
        Blocklength allocated to the corresponding UE
    rate : float or numpy.ndarray
        Coding rate (r = d/m, where d is the data size)
    
    Returns:
    --------
    err : float or numpy.ndarray
        Error probability
    """
    # Calculate channel dispersion
    V = 1 - 1 / (snr + 1)**2
    
    # Calculate the normalized decoding error probability
    w = (blocklength / V)**0.5 * (np.log2(1 + snr) - rate)
    
    # Handle cases with imaginary results (due to numerical issues)
    if isinstance(w, np.ndarray):
        w[np.iscomplex(w)] = -np.inf
        w[snr <= 0] = -np.inf
    else:
        if np.iscomplex(w) or snr <= 0:
            w = -np.inf
    
    # Calculate the error probability using Q-function (via complementary CDF of normal distribution)
    err = norm.cdf(-w)
    return err

def solve_power_allocation_cvxpy(h_i, weights, later_weights, data_size, blocklength, 
                                N0, B, P_max, P_sum, theta, Tslot):
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
        Data size for each user
    blocklength : numpy.ndarray
        Blocklength allocated to each user
    N0 : float
        Noise power spectral density
    B : float
        Bandwidth
    P_max : float
        Maximum power per user
    P_sum : float
        Sum power constraint
    theta : float
        Energy coefficient related to computation
    Tslot : float
        Time slot duration
        
    Returns:
    --------
    P_opt : numpy.ndarray
        Optimal power allocation
    obj_value : float
        Objective function value at the optimal solution
    """
    K = len(h_i)
    
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
    
    # Basic constraints
    constraints = [
        P >= 0,                                                  # Power must be non-negative
        P <= P_max,                                              # Power per user must not exceed maximum
        cp.sum(P) <= P_sum,    # Total power constraint
    ]
    
    # Add constraints to ensure error probabilities are between 0 and 1
    for i in range(K):
        # Constraint to ensure error probability is at least 0
        # 0.5 - (mu_values[i] / np.sqrt(2 * np.pi)) * (P[i] / D_i[i] - alpha_values[i]) >= 0
        constraints.append(0.5 - (mu_values[i] / np.sqrt(2 * np.pi)) * (P[i] / D_i[i] - alpha_values[i]) >= 0)
        
        # Constraint to ensure error probability is at most 1
        # 0.5 - (mu_values[i] / np.sqrt(2 * np.pi)) * (P[i] / D_i[i] - alpha_values[i]) <= 1
        #constraints.append(0.5 - (mu_values[i] / np.sqrt(2 * np.pi)) * (P[i] / D_i[i] - alpha_values[i]) <= 1)
    
    # Define the error probability expression for each user
    err_prob_expressions = []
    for i in range(K):
        # Calculate error probability expression (without using cp.maximum/cp.minimum since we have explicit constraints)
        err_prob_i = 0.5 - (mu_values[i] / np.sqrt(2 * np.pi)) * (P[i] / D_i[i] - alpha_values[i])
        err_prob_expressions.append(err_prob_i)
    
    # Convert the list of expressions to a vector expression using vstack
    err_prob_vector = cp.vstack(err_prob_expressions)
    
    # Objective function with error probability as a function of P
    objective_expression = cp.sum(cp.multiply(weights, err_prob_vector)) - cp.sum(later_weights)
    objective = cp.Minimize(objective_expression)
    
    # Create and solve the problem
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.MOSEK)  # Use MOSEK if available
    except:
        try:
            prob.solve(solver=cp.ECOS)  # Fallback to ECOS
        except:
            prob.solve()  # Let CVXPY choose the solver
    
    # Check if the problem was solved successfully
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Problem could not be solved optimally. Status: {prob.status}")
        return None, float('inf')
    
    return P.value, prob.value

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
    
    # Calculate nu parameter (though not used in the final calculation)
    nu = np.sqrt(np.pi / (2 * mu**2))
    
    # Calculate error approximation
    err_approx = 0.5 - (mu / np.sqrt(2 * np.pi)) * (snr - alpha)
    
    # # Handle cases where approximation might be out of [0,1] bounds
    # if isinstance(err_approx, np.ndarray):
    #     err_approx = np.clip(err_approx, 0, 1)
    # else:
    #     err_approx = max(0, min(1, err_approx))
    
    return err_approx

def main():
    parser = argparse.ArgumentParser(description='Solve power allocation problem using CVXPY')
    parser.add_argument('--K', type=int, default=3, help='Number of users')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--P_max', type=float, default=10, help='Maximum power per user (mW)')
    parser.add_argument('--P_sum', type=float, default=60, help='Sum power constraint (mW)')
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Initialize wireless parameters
    K = args.K
    wireless_arg = {
        'radius': 1000,
        'ampli': 15,
        'N0': dB2power(-150),
        'B': 1e6,
        'm': dB2power(0.023),
        'M': 16,
        'Mprime': 15,
        'E_max': 60,  # mJ
        'Tslot': 1.3,
        'sigma': 1,
        'freq': 2400,  # MHz
        'P_max': args.P_max,  # mW
        'alpha': 0.1,
        'beta': 0.001,
        'kappa': 10**(-28),
        'freq_comp': 2*10**9,
        'C': 2*10**4
    }
    
    # Generate user distances
    distance = np.sqrt(np.random.uniform(1, wireless_arg['radius']**2, K))
    
    # Calculate path loss
    FSPL = 20 * np.log10(distance) + 20 * np.log10(wireless_arg['freq']) - 27.55
    
    # Generate channel gains (Rayleigh fading)
    o_i = wireless_arg['sigma'] * np.sqrt(np.square(np.random.randn(K)) + np.square(np.random.randn(K)))
    h_i = o_i / dB2power(FSPL)
    
    # Generate weights, data sizes, and blocklengths
    weights = np.ones(K)  # Equal weights
    later_weights = np.ones(K) * 0.1  # Small random values for later weights
    data_size = np.ones(K) * 19  # 700 bits per user
    blocklength = np.ones(K) * 20  # 500 channel uses per user
    
    # Calculate theta
    theta = wireless_arg['kappa'] * wireless_arg['freq_comp']**2 * wireless_arg['C'] * 20 * 60000
    
    # Print parameter table
    print("\n=== Optimization Parameters ===")
    print(f"{'Parameter':<20} {'Value':<15}")
    print("-" * 35)
    print(f"{'N0':<20} {wireless_arg['N0']:<15.6e}")
    print(f"{'B':<20} {wireless_arg['B']:<15.0f}")
    print(f"{'P_max':<20} {wireless_arg['P_max']:<15.2f}")
    print(f"{'P_sum':<20} {args.P_sum:<15.2f}")
    print(f"{'theta':<20} {theta:<15.6e}")
    print(f"{'Tslot':<20} {wireless_arg['Tslot']:<15.2f}")
    print(f"{'Using':<20} {'FBL Approximation':<15}")
    
    # Print channel gains for users
    print("\n=== Channel Gains ===")
    print(f"{'UE Index':<10} {'h_i':<15} {'Distance':<15} {'SNR @ max power (dB)':<20}")
    print("-" * 60)
    
    for i in range(K):
        max_snr = args.P_max * h_i[i] / (wireless_arg['N0'] * wireless_arg['B'])
        max_snr_db = 10 * np.log10(max_snr) if max_snr > 0 else float('-inf')
        
        print(f"{i:<10} {h_i[i]:<15.6e} {distance[i]:<15.2f} {max_snr_db:<20.2f}")
    
    # Solve the power allocation problem
    P_opt, obj_value = solve_power_allocation_cvxpy(
        h_i, weights, later_weights, data_size, blocklength,
        wireless_arg['N0'], wireless_arg['B'], args.P_max, args.P_sum,
        theta, wireless_arg['Tslot']
    )
    
    if P_opt is not None:
        # Calculate SNRs and error probabilities
        D_i = wireless_arg['N0'] * wireless_arg['B'] / h_i
        snr_opt = P_opt / D_i
        
        # Use the approximation method for error probability calculation
        err_prob = error_prob_fbl_approx(snr_opt, blocklength, data_size)
        
        # Print optimization results
        print("\n=== Optimization Results ===")
        print(f"Objective function value: {obj_value:.6f}")
        
        print("\n=== Optimal Power Allocation ===")
        print(f"{'UE Index':<10} {'Power (mW)':<15} {'SNR (dB)':<15} {'Error Prob':<15}")
        print("-" * 60)
        
        for i in range(K):
            snr_db = 10 * np.log10(snr_opt[i]) if snr_opt[i] > 0 else float('-inf')
            print(f"{i:<10} {P_opt[i]:<15.6f} {snr_db:<15.2f} {err_prob[i]:<15.6f}")
        
        # Calculate total power consumption
        total_power = np.sum(P_opt) 
        print(f"\nTotal power consumption: {total_power:.6f} / {args.P_sum:.6f}")
        print("\n=== debugging error probability ===")
        err_prob_debug=error_prob_fbl_approx(1, 600, 400)
        print(err_prob_debug)
if __name__ == "__main__":
    main()