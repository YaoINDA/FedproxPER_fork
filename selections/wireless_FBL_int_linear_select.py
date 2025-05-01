import numpy as np
import copy
import torch
from scipy.stats import norm

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

def LR_selection_FBL(args, wireless_arg, seed, data_size, weights, later_weights, blocklength=None):
    """
    Linear user selection algorithm for finite blocklength communication model.
    
    This algorithm:
    1. Calculates the maximum power needed for each user
    2. Calculates the gain for each user
    3. Selects users in decreasing order of gain until power budget is exhausted
    
    Parameters:
    -----------
    args : argparse.Namespace
        Configuration parameters
    wireless_arg : dict
        Wireless communication parameters
    seed : int
        Random seed for reproducibility
    data_size : numpy.ndarray
        Data size for each user
    weights : numpy.ndarray
        User weights
    later_weights : numpy.ndarray
        Additional user weights
    blocklength : int or numpy.ndarray, optional
        Blocklength allocated to each user
        
    Returns:
    --------
    active_success_clients : list
        List of selected users
    proba_success_avg : numpy.ndarray
        Success probability for each user (set to 1)
    fails : int
        Number of users that were not selected
    success_rate : float
        Ratio of selected users to target users
    obj_value : float
        Sum of gains for selected users
    """
    print("\n=== LR_selection_FBL DIAGNOSTICS ===")
    print(f"Seed: {seed}")
    
    # Initialize parameters
    user_indices = [k for k in range(args.total_UE)]
    h_i = copy.deepcopy(wireless_arg['h_i'])
    h_avg = copy.deepcopy(wireless_arg['h_avg'])
    N0 = wireless_arg['N0']
    B = wireless_arg['B']
    P_max = wireless_arg['P_max']
    P_sum = wireless_arg['P_sum']
    K = args.active_UE
    
    # Handle blocklength - ensure it's an array
    if blocklength is None:
        blocklength = 500  # Default blocklength
        print(f"Using default blocklength: {blocklength}")
    
    # Convert blocklength to array if it's a scalar
    if np.isscalar(blocklength):
        blocklength_array = np.ones(args.total_UE) * blocklength
    else:
        blocklength_array = blocklength
        
    # Get packet size
    if 'Packet_size' in wireless_arg:
        packet_size = np.ones(args.total_UE) * wireless_arg['Packet_size']
    else:
        packet_size = np.ones(args.total_UE) * (args.packet_size if hasattr(args, 'packet_size') else 500)
    
    # Calculate alpha and mu parameters for each user
    alpha_values = np.zeros(args.total_UE)
    mu_values = np.zeros(args.total_UE)
    
    for i in range(args.total_UE):
        alpha_values[i] = np.exp(packet_size[i] / blocklength_array[i]) - 1
        mu_values[i] = np.sqrt(blocklength_array[i] / (np.exp(2 * packet_size[i] / blocklength_array[i]) - 1))
    
    # Calculate P_err_max_unbound for each user
    P_err_max_unbound = N0 * B / h_i * (np.sqrt(2 * np.pi) / (2 * mu_values) + alpha_values)
    
    # Calculate P_err_max for each user
    P_err_max = np.minimum(P_err_max_unbound, P_max)
    
    # Calculate gain for each user
    snr_max = P_err_max * h_i / (N0 * B)
    gain = weights * (0.5 - (mu_values / np.sqrt(2 * np.pi)) * (snr_max - alpha_values))
    
    # Add later_weights to gain
    gain = gain + later_weights
    
    print(f"Number of users: {len(user_indices)}")
    print(f"P_max: {P_max}, P_sum: {P_sum}")
    print(f"Target number of users (K): {K}")
    
    # Iterative selection process
    P_left = P_sum
    active_clients = []
    gains_selected = []
    
    # Sort users by gain in descending order
    sorted_indices = np.argsort(-gain)
    
    for idx in sorted_indices:
        if P_left >= P_err_max[idx] and len(active_clients) < K:
            active_clients.append(idx)
            gains_selected.append(gain[idx])
            P_left -= P_err_max[idx]
            print(f"Selected user {idx}, gain: {gain[idx]:.4f}, power: {P_err_max[idx]:.4f}, P_left: {P_left:.4f}")
        else:
            # If we can't select this user, continue to the next one
            continue
    
    K_select = len(active_clients)
    print(f"Selected {K_select} users out of {K}")
    
    # Calculate return values
    proba_success_avg = np.ones(args.total_UE)  # Always 1 as specified
    fails = K - K_select
    success_rate = K_select / K if K > 0 else 0
    obj_value = sum(gains_selected)
    
    print(f"Objective value (sum of gains): {obj_value:.4f}")
    print(f"Success rate: {success_rate:.4f}")
    
    return active_clients, proba_success_avg, fails, success_rate, obj_value

def user_selection_fbl_int_linear(args, wireless_arg, seed, data_size, weights, later_weights, blocklength=None):
    """
    Wrapper function that calls LR_selection_FBL and handles any errors.
    
    Parameters are the same as LR_selection_FBL.
    """
    try:
        return LR_selection_FBL(args, wireless_arg, seed, data_size, weights, later_weights, blocklength)
    except Exception as e:
        print(f"Error in LR_selection_FBL: {e}")
        import traceback
        traceback.print_exc()
        
        # Fall back to random selection
        np.random.seed(seed)
        active_clients = np.random.choice(range(args.total_UE), min(args.active_UE, args.total_UE), replace=False)
        print(f"Falling back to random selection: {active_clients}")
        
        return list(active_clients), np.ones(args.total_UE), args.active_UE - len(active_clients), len(active_clients)/args.active_UE, 0