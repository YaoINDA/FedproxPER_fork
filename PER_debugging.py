import numpy as np
import torch
from utils.options import args_parser
from selections.wireless import dB2power, wireless_param, update_wireless
from selections.function_user_selection_allCombinations import solve_opti_wrt_P, error_prob_fbl_approx

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
    
    # Select the first 3 UEs
    K = 4
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
    data_size = np.ones(K) * 700  # 700 bits per UE
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
    
    # Test different power allocations to see their objective values
    print("\n=== Testing Different Power Allocations ===")
    test_powers = [
        np.ones(K) * 0.1,  # Very low power
        np.ones(K) * 1.0,   # Medium power
        np.ones(K) * P_max  # Maximum power
    ]
    
    for i, power in enumerate(test_powers):
        # Calculate SNR values
        D_i = m*N0*B/h_i
        snr = power/D_i
        
        # Calculate error probabilities (approximated FBL)
        err_prob_app = error_prob_fbl_approx(snr, blocklength, data_size)
        # Set error probability to 1 for SNR <= 0
        mask = snr <= 0
        err_prob_app[mask] = 1.0
        
        # Calculate objective value for the approximated model
        obj_val_approx = np.sum(weights * err_prob_app) - np.sum(later_weights)
        
        print(f"\nPower allocation {i+1}: {power}")
        print(f"SNR values: {snr}")
        print(f"SNR (dB): {10*np.log10(snr)}")
        print(f"Error probabilities (approx FBL): {err_prob_app}")
        print(f"Objective value (approx FBL): {obj_val_approx:.6f}")
    
    # Solve the optimization problem using approximated FBL error model
    print("\n=== Solving Optimization Problem with Approximated FBL Error Model ===")
    try:
        obj_value_approx, opt_power_approx = solve_opti_wrt_P(
            weights, h_i, N0, B, m, K, 
            wireless_arg['alpha'], wireless_arg['beta'],
            P_max, P_sum, 'P_FBL_app', data_size, theta, Tslot, 
            later_weights, blocklength=blocklength
        )
        
        # Print optimization results
        print("\n=== Optimization Results (Approximated FBL) ===")
        print(f"Objective function value: {obj_value_approx:.6f}")  
        
        # Calculate SNR and error probabilities for verification
        D_i = m*N0*B/h_i
        snr_opt_approx = opt_power_approx/D_i
        err_prob_opt_approx = error_prob_fbl_approx(snr_opt_approx, blocklength, data_size)
        mask = snr_opt_approx <= 0
        err_prob_opt_approx[mask] = 1.0
        
        # Manually calculate the objective value
        obj_manual_approx = np.sum(weights * err_prob_opt_approx) - np.sum(later_weights)
        print(f"Manually calculated objective: {obj_manual_approx:.6f}")
        
        print("\n=== Optimal Power Allocation (Approximated FBL) ===")
        print(f"{'UE Index':<10} {'Power (mW)':<15} {'SNR (dB)':<15} {'Error Prob':<15}")
        print("-" * 60)
        
        for i, ue_idx in enumerate(user_indices):
            snr = opt_power_approx[i] * h_i[i] / (N0 * B)
            snr_db = 10 * np.log10(snr) if snr > 0 else float('-inf')
            error_prob = err_prob_opt_approx[i]
            
            print(f"{ue_idx:<10} {opt_power_approx[i]:<15.6f} {snr_db:<15.2f} {error_prob:<15.6f}")
        
        # Calculate total power
        total_power_approx = np.sum(opt_power_approx) + theta/Tslot * np.sum(data_size)
        print(f"\nTotal power consumption: {total_power_approx:.6f} / {P_sum:.6f}")
        
    except Exception as e:
        print(f"Error in approximated FBL optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_fbl_optimization()