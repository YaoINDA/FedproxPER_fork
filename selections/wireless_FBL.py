import numpy as np
import copy
import torch
from scipy.stats import norm
from selections.function_power_allocation_FBL_approx import power_allocation_fbl_approx, error_prob_fbl_approx
def error_prob_fbl(snr, blocklength, rate):
    """
    Calculate the error probability for finite blocklength communications.
    """
    # Handle scalar inputs
    scalar_input = np.isscalar(snr)
    if scalar_input:
        snr = np.array([snr])
        blocklength = np.array([blocklength])
        rate = np.array([rate])
    
    # Calculate channel dispersion
    V = 1 - 1 / (snr + 1)**2
    
    # Calculate the normalized decoding error probability
    w = (blocklength / V)**0.5 * (np.log2(1 + snr) - rate)
    
    # Handle cases with imaginary results (due to numerical issues)
    w[np.iscomplex(w)] = -np.inf
    w[snr <= 0] = -np.inf
    
    # Calculate the error probability using Q-function
    err = norm.cdf(-w)
    
    # Return scalar if input was scalar
    if scalar_input:
        return err[0]
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
    # Add detailed logging
    print("\n=== FBL USER SELECTION DIAGNOSTICS ===")
    print(f"Seed: {seed}")
    
    # Initialize parameters
    user_indices = [k for k in range(args.total_UE)]
    h_i = copy.deepcopy(wireless_arg['h_i'])
    h_avg = copy.deepcopy(wireless_arg['h_avg'])
    N0 = wireless_arg['N0']
    B = wireless_arg['B']
    m = wireless_arg['m']
    const_alpha = N0 * B / m
    P_max = wireless_arg['P_max']
    P_sum = wireless_arg['P_sum']
    theta = wireless_arg['theta']
    Tslot = wireless_arg['Tslot']
    K = args.active_UE
    
    # Log key parameters
    print("\n=== Key Parameters ===")
    print(f"N0: {N0}")
    print(f"B: {B}")
    print(f"P_max: {P_max}")
    print(f"P_sum: {P_sum}")
    print(f"theta: {theta}")
    print(f"Tslot: {Tslot}")
    
    # Handle blocklength - ensure it's an array
    if blocklength is None:
        blocklength = 700  # Default blocklength
        print(f"Using default blocklength: {blocklength}")
    print(f"Blocklength type: {type(blocklength)}")
    
    # Convert blocklength to array if it's a scalar
    if np.isscalar(blocklength):
        blocklength_array = np.ones(args.total_UE) * blocklength
        print(f"Converted blocklength to array with value: {blocklength}")
    else:
        blocklength_array = blocklength
        print(f"Using blocklength array with shape: {np.shape(blocklength_array)}")
    
    # Get packet size
    if 'Packet_size' in wireless_arg:
        packet_size = np.ones(args.total_UE) * wireless_arg['Packet_size']
        print(f"Using packet_size from wireless_arg: {wireless_arg['Packet_size']}")
    else:
        packet_size = np.ones(args.total_UE) * (args.packet_size if hasattr(args, 'packet_size') else 500)
        print(f"Using packet_size from args or default: {packet_size[0]}")

    # Handle data_size - ensure it's an array
    if np.isscalar(data_size):
        data_size_array = np.ones(args.total_UE) * data_size
        print(f"Converted data_size scalar to array: {data_size}")
    else:
        data_size_array = data_size
        print(f"Using data_size array with shape: {np.shape(data_size_array)}")

    # User selection strategies
    
    if args.selection == 'uni_random':
        print("Selection method: uniform random")
        np.random.seed(seed)
        active_clients = np.random.choice(user_indices, args.active_UE, replace=False)
    elif args.selection == 'best_channel':
        print("Selection method: best channel")
        active_clients = np.argsort(-h_avg)[0:args.active_UE]
    elif args.selection == 'best_channel_ratio':
        print("Selection method: best channel ratio")
        active_clients = np.argsort(-h_i / wireless_arg['h_avg'])[0:args.active_UE]
    elif args.selection == 'best_loss':
        print("Selection method: best loss")
        active_clients = np.argsort(-weights)[0:args.active_UE]
    elif args.selection == 'weighted_random':
        print("Selection method: weighted random")
        torch.manual_seed(seed)
        # Use data_size_array for weighted random sampling
        active_clients = list(torch.utils.data.WeightedRandomSampler(data_size_array, args.active_UE, replacement=False))
    else:
        print(f"Unknown selection strategy: {args.selection}, using uniform random")
        np.random.seed(seed)
        active_clients = np.random.choice(user_indices, args.active_UE, replace=False)

    print(f"Selected clients: {active_clients}")

    # Extract parameters for selected users
    h_i_p = copy.deepcopy(h_i[active_clients])
    packet_size_p = copy.deepcopy(packet_size[active_clients])
    later_weights_p = copy.deepcopy(later_weights[active_clients])
    weights_p = copy.deepcopy(weights[active_clients])
    blocklength_p = copy.deepcopy(blocklength_array[active_clients])
    
    print(f"Channel gains for selected users: {h_i_p}")
    print(f"Packet sizes for selected users: {packet_size_p}")
    print(f"Blocklengths for selected users: {blocklength_p}")
    
    # Allocate power using FBL approximation
    power_allocated = np.zeros(len(user_indices))
    
    try:
        print("\n=== Calling power_allocation_fbl_approx ===")
        P_opt, obj_val, status = power_allocation_fbl_approx(
            h_i_p, weights_p, later_weights_p, packet_size_p, blocklength_p,
            N0, B, P_max, P_sum, theta, Tslot
        )
        
        print(f"Optimization status: {status}")
        print(f"Objective value: {obj_val}")

        if P_opt is not None:
            print(f"Power allocation: {P_opt}")
            power_allocated[active_clients] = P_opt
            # Calculate error probabilities 
            snr = P_opt * h_i_p / (N0 * B)
            err_prob_selected = error_prob_fbl_approx(snr, blocklength_p, packet_size_p)
            print(f"Error probabilities: {err_prob_selected}")
        else:
            print("P_opt is None, using uniform power allocation")
            power_allocated[active_clients] = np.ones(len(active_clients)) * min(P_max, P_sum / len(active_clients))
    except Exception as e:
        print(f"Error in FBL power allocation: {e}")
        import traceback
        traceback.print_exc()
        print("Exception occurred, using uniform power allocation")
        power_allocated[active_clients] = np.ones(len(active_clients)) * min(P_max, P_sum / len(active_clients))

    # Calculate objective value based on FBL error model (using arrays for all parameters)
    obj_value = objective_function_fbl(power_allocated, h_i, weights, packet_size, blocklength_array, N0, B, later_weights)
    
    # Calculate SNR for each user
    snr = power_allocated * h_i / (N0 * B)
    
    # Calculate rate for each user (bits per channel use)
    rate = packet_size / blocklength_array
    
    # Calculate error probability for each user
    err_prob = np.ones(args.total_UE)  # Default to 1 (failure)
    mask = snr > 0
    
    if np.any(mask):
        # Use blocklength_array to ensure it's an array that can be indexed
        err_prob[mask] = error_prob_fbl(snr[mask], blocklength_array[mask], rate[mask])
    
    # Calculate success probability (complement of error probability)
    proba_success = 1 - err_prob
    
    # Calculate average success probability based on average channel
    snr_avg = power_allocated * h_avg / (N0 * B)
    proba_success_avg = np.zeros(args.total_UE)
    mask_avg = snr_avg > 0
    
    if np.any(mask_avg):
        # Use blocklength_array to ensure it's an array that can be indexed
        rate_avg = packet_size[mask_avg] / blocklength_array[mask_avg]
        err_prob_avg = error_prob_fbl(snr_avg[mask_avg], blocklength_array[mask_avg], rate_avg)
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
    print(f"Successfully transmitted clients: {active_success_clients}")

    return list(set(active_success_clients)), proba_success_avg, fails, 1-avg_err_prob, obj_value