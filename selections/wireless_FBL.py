#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy
import torch
from scipy.stats import norm
from selections.wireless import dB2power, wireless_param, update_wireless
from selections.function_power_allocation_FBL_approx import power_allocation_fbl_approx

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

def objective_function_fbl(power, h_i, weights, packet_size, blocklength, N0, B, later_weights):
    """
    Calculate objective function value based on FBL error probability model.
    
    Parameters:
    -----------
    power : numpy.ndarray
        Power allocation for each user
    h_i : numpy.ndarray
        Channel gains for each user
    weights : numpy.ndarray
        Weights for each user
    packet_size : numpy.ndarray
        Packet size for each user
    blocklength : numpy.ndarray or float
        Blocklength allocated to each user
    N0 : float
        Noise power spectral density
    B : float
        Bandwidth
    later_weights : numpy.ndarray
        Additional weights for each user
        
    Returns:
    --------
    obj_value : float
        Objective function value
    """
    # Ensure blocklength is an array
    if np.isscalar(blocklength):
        blocklength = np.ones_like(power) * blocklength
    
    # Calculate SNR for each user
    snr = np.zeros_like(power)
    mask = power > 0
    snr[mask] = power[mask] * h_i[mask] / (N0 * B)
    
    # Calculate rate for each user (bits per channel use)
    rate = packet_size / blocklength
    
    # Calculate error probability using FBL model
    err_prob = np.ones_like(power)  # Default to 1 (failure)
    if np.any(mask):
        err_prob[mask] = error_prob_fbl(snr[mask], blocklength[mask], rate[mask])
    
    # Calculate objective function value
    return np.sum(weights * err_prob) - np.sum(later_weights)

def user_selection_fbl(args, wireless_arg, seed, data_size, weights, later_weights, blocklength=None):
    """
    Select users and allocate power using FBL error probability model.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    wireless_arg : dict
        Dictionary of wireless parameters
    seed : int
        Random seed for selection
    data_size : numpy.ndarray
        Array of data sizes for each user
    weights : numpy.ndarray
        Weight for each user in the objective function
    later_weights : numpy.ndarray
        Additional weights for each user
    blocklength : int or numpy.ndarray, optional
        Blocklength allocated to each user or single value for all users
        
    Returns:
    --------
    active_success_clients : list
        List of successfully transmitted client indices
    proba_success_avg : numpy.ndarray
        Average success probability for each client
    fails : int
        Number of failed transmissions
    avg_err_prob : float
        Average error probability for active clients
    obj_value : float
        Objective function value
    """
    # Initialize parameters
    user_indices = [k for k in range(args.total_UE)]
    h_i = copy.deepcopy(wireless_arg['h_i'])
    h_avg = copy.deepcopy(wireless_arg['h_avg'])
    N0 = wireless_arg['N0']
    B = wireless_arg['B']
    m = wireless_arg['m']
    const_alpha = N0 * B / m  # For compatibility with original objective function
    P_max = wireless_arg['P_max']
    P_sum = wireless_arg['P_sum']
    theta = wireless_arg['theta']
    Tslot = wireless_arg['Tslot']
    K = args.active_UE
    
    # Set default blocklength if not provided
    if blocklength is None:
        blocklength = 700  # Default blocklength
    
    # Get packet size (corrected: use wireless_arg['Packet_size'] instead of data_size)
    if 'Packet_size' in wireless_arg:
        packet_size = np.ones(args.total_UE) * wireless_arg['Packet_size']
    else:
        # Fallback if Packet_size is not defined in wireless_arg
        packet_size = np.ones(args.total_UE) * args.packet_size if hasattr(args, 'packet_size') else np.ones(args.total_UE) * 500

    # User selection strategies (similar to wireless.py)
    if args.selection == 'uni_random':
        np.random.seed(seed)
        active_clients = np.random.choice(user_indices, args.active_UE, replace=False)
    elif args.selection == 'best_channel':
        active_clients = np.argsort(-h_avg)[0:args.active_UE]
    elif args.selection == 'best_channel_ratio':
        active_clients = np.argsort(-h_i / wireless_arg['h_avg'])[0:args.active_UE]
    elif args.selection == 'best_loss':
        active_clients = np.argsort(-weights)[0:args.active_UE]
    elif args.selection == 'weighted_random':
        torch.manual_seed(seed)
        active_clients = list(torch.utils.data.WeightedRandomSampler(data_size, args.active_UE, replacement=False))
    else:
        print("Unknown selection strategy, using uniform random selection")
        np.random.seed(seed)
        active_clients = np.random.choice(user_indices, args.active_UE, replace=False)

    # Extract parameters for selected users
    h_i_p = copy.deepcopy(h_i[active_clients])
    packet_size_p = copy.deepcopy(packet_size[active_clients])
    later_weights_p = copy.deepcopy(later_weights[active_clients])
    weights_p = copy.deepcopy(weights[active_clients])
    
    # Ensure blocklength is an array with proper dimensions
    if np.isscalar(blocklength):
        blocklength_p = np.ones(len(active_clients)) * blocklength
    else:
        blocklength_p = copy.deepcopy(blocklength[active_clients])
    
    # Allocate power using FBL approximation
    power_allocated = np.zeros(len(user_indices))
    
    try:
        # Use power allocation with FBL approximation
        P_opt, obj_val, status = power_allocation_fbl_approx(
            h_i_p, weights_p, later_weights_p, packet_size_p, blocklength_p,
            N0, B, P_max, P_sum, theta, Tslot
        )

        if P_opt is not None:
            power_allocated[active_clients] = P_opt
        else:
            # Fallback: use uniform power allocation
            power_allocated[active_clients] = np.ones(len(active_clients)) * min(P_max, P_sum / len(active_clients))
            print("FBL power allocation failed, using uniform power allocation")
    except Exception as e:
        print(f"Error in FBL power allocation: {e}")
        # Fallback to uniform power allocation
        power_allocated[active_clients] = np.ones(len(active_clients)) * min(P_max, P_sum / len(active_clients))

    # Calculate error probability using FBL model
    if np.isscalar(blocklength):
        bl = np.ones(args.total_UE) * blocklength
    else:
        bl = copy.deepcopy(blocklength)  

    # Calculate objective value based on FBL error model (using packet_size instead of data_size)
    obj_value = objective_function_fbl(power_allocated, h_i, weights, packet_size, bl, N0, B, later_weights)
    
    # Calculate SNR for each user
    snr = power_allocated * h_i / N0 / B
    
    # Calculate rate for each user (bits per channel use)
    rate = packet_size / bl
    
    # Calculate error probability for each user
    err_prob = np.ones(args.total_UE)  # Default to 1 (failure)
    mask = snr > 0
    
    if np.any(mask):
        err_prob[mask] = error_prob_fbl(snr[mask], bl[mask], rate[mask])
    
    # Calculate success probability (complement of error probability)
    proba_success = 1 - err_prob
    
    # Calculate average success probability based on average channel
    snr_avg = power_allocated * h_avg / N0 / B
    proba_success_avg = np.zeros(args.total_UE)
    mask_avg = snr_avg > 0
    
    if np.any(mask_avg):
        # Use packet_size instead of data_size for rate calculation
        rate_avg = packet_size[mask_avg] / bl[mask_avg]
        err_prob_avg = error_prob_fbl(snr_avg[mask_avg], bl[mask_avg], rate_avg)
        proba_success_avg[mask_avg] = 1 - err_prob_avg
    
    # Simulate transmission success/failure
    active_success_clients = []
    fails = 0
    np.random.seed(seed+123)
    
    for idx, client in enumerate(active_clients):
        success = np.random.binomial(1, proba_success[client])
        if success:
            active_success_clients.append(client)
        else:
            fails += 1
    
    # Calculate average error probability for active clients
    avg_err_prob = np.mean(err_prob[active_clients]) if len(active_clients) > 0 else 1.0
    
    print(f"Number of selected users: {len(active_clients)}")
    print(f"Number of failed transmissions: {fails}")
    print(f"Average error probability: {avg_err_prob:.4f}")

    # Debugging information
    cnr_i = h_i / N0 / B
    cnr_active = cnr_i[mask]
    print(f"Power allocation status: {status if 'status' in locals() else 'N/A'}")
    print(f"Power max: {P_max}, Power sum: {P_sum}")
    print(f"h_i: {h_i_p}")
    print(f"optimal power: {P_opt if 'P_opt' in locals() else 'N/A'}")
    print(f"SNR: {snr[mask]}")
    print(f"CNR: {cnr_active}")
    print(f"blockength: {blocklength_p}")
    print(f"packet_size_p: {packet_size_p}")  # Changed from data_size_p to packet_size_p
    
    return list(set(active_success_clients)), proba_success_avg, fails, 1-avg_err_prob, obj_value