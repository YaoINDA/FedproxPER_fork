import numpy as np
import torch
from utils.options import args_parser
from selections.wireless import dB2power, wireless_param, update_wireless, error_prob_fbl
from selections.function_user_selection_allCombinations import solve_opti_wrt_P

def debug_fbl_optimization():
    # Parse arguments with default values
    args = args_parser()
    print("=== Configuration ===")
    print(f"Total UEs: {args.total_UE}")
    print(f"Active UEs: {args.active_UE}")
    print(f"Random seed: {args.wireless_seed}")
    
    # Initialize dummy data weights for wireless param initialization
    data_weight = np.ones(args.total_UE) / args.total_UE
    nb_data_assigned = args.total_UE  # Dummy value
    
    # Initialize and update wireless parameters
    wireless_arg = wireless_param(args, data_weight, nb_data_assigned)
    wireless_arg = update_wireless(args, wireless_arg, args.wireless_seed)
    
    # Select the first 10 UEs
    K = 10
    user_indices = np.arange(K)
    print(f"\n=== Selected UEs: {user_indices} ===")
    
    # Extract parameters for the selected UEs
    h_i = wireless_arg['h_i'][user_indices]
    h_avg = wireless_arg['h_avg'][user_indices]
    N0 = wireless_arg['N0']
    B = wireless_arg['B']
    m = wireless_arg['m']
    P_max = wireless_arg['P_max']
    P_sum = wireless_arg['P_sum']
    theta = wireless_arg['theta']
    Tslot = wireless_arg['Tslot']
    
    # For FBL error model, we need data size and blocklength
    data_size = np.ones(K) * 1000  # 1000 bits per UE
    weights = np.ones(K)  # Equal weights
    later_weights = np.zeros(K)  # No later weights
    blocklength = np.ones(K) * 500  # 500 channel uses per UE
    
    # Print parameter table
    print("\n=== Optimization Parameters ===")
    print(f"{'Parameter':<20} {'Value':<15}")
    print("-" * 35)
    print(f"{'N0':<20} {N0:<15.6e}")
    print(f"{'B':<20} {B:<15.0f}")
    print(f"{'m':<20} {m:<15.6f}")
    print(f"{'P_max':<20} {P_max:<15.2f}")
    print(f"{'P_sum':<20} {P_sum:<15.2f}")
    print(f"{'theta':<20} {theta:<15.6e}")
    print(f"{'Tslot':<20} {Tslot:<15.2f}")
    
    # Print channel gains for selected UEs
    print("\n=== Channel Gains ===")
    print(f"{'UE Index':<10} {'h_i':<15} {'Distance':<15} {'SNR @ max power (dB)':<20}")
    print("-" * 60)
    
    for i, ue_idx in enumerate(user_indices):
        distance = wireless_arg['distance'][ue_idx]
        max_snr = P_max * h_i[i] / (N0 * B)
        max_snr_db = 10 * np.log10(max_snr) if max_snr > 0 else float('-inf')
        
        print(f"{ue_idx:<10} {h_i[i]:<15.6e} {distance:<15.2f} {max_snr_db:<20.2f}")
    
    # Solve the optimization problem using FBL error model
    print("\n=== Solving Optimization Problem with FBL Error Model ===")
    try:
        obj_value, opt_power = solve_opti_wrt_P(
            weights, h_i, N0, B, m, K, 
            wireless_arg['alpha'], wireless_arg['beta'],
            P_max, P_sum, 'P_FBL', data_size, theta, Tslot, 
            later_weights, blocklength
        )
        
        # Print optimization results
        print("\n=== Optimization Results ===")
        print(f"Objective function value: {-obj_value:.6f}")
        
        print("\n=== Optimal Power Allocation ===")
        print(f"{'UE Index':<10} {'Power (mW)':<15} {'SNR (dB)':<15} {'Error Prob':<15}")
        print("-" * 60)
        
        for i, ue_idx in enumerate(user_indices):
            snr = opt_power[i] * h_i[i] / (N0 * B)
            snr_db = 10 * np.log10(snr) if snr > 0 else float('-inf')
            error_prob = error_prob_fbl(snr, blocklength[i], data_size[i]/blocklength[i])
            
            print(f"{ue_idx:<10} {opt_power[i]:<15.6f} {snr_db:<15.2f} {error_prob:<15.6f}")
        
        # Calculate total power
        total_power = np.sum(opt_power) + theta/Tslot * np.sum(data_size)
        print(f"\nTotal power consumption: {total_power:.6f} / {P_sum:.6f}")
        
    except Exception as e:
        print(f"Error in optimization: {e}")
        
if __name__ == "__main__":
    debug_fbl_optimization()