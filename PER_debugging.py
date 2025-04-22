import numpy as np
import torch
from utils.options import args_parser
from selections.wireless import dB2power, wireless_param, update_wireless
from selections.function_power_allocation_FBL_approx import power_allocation_fbl_approx, error_prob_fbl_approx
from scipy.stats import norm  # For error_prob_fbl function

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

def debug_fbl_optimization():
    # Parse arguments with default values
    args = args_parser()
    args.total_UE = 30  # Ensure we have at least 10 UEs to select
    args.active_UE = 10  # We will select 10 UEs
    
    print("=== Configuration ===")
    print(f"Total UEs: {args.total_UE}")
    print(f"Active UEs: {args.active_UE}")
    print(f"Random seed: {args.wireless_seed}")
    
    # Initialize dummy data weights for wireless param initialization
    data_weight = np.ones(args.total_UE) / args.total_UE
    nb_data_assigned = args.total_UE  # Dummy value
    
    # Initialize and update wireless parameters
    wireless_arg = wireless_param(args, data_weight, nb_data_assigned)
    wireless_arg = update_wireless(args, wireless_arg, args.wireless_seed+1)
    
    # Select the first 10 UEs
    K = 5
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
    data_size = np.ones(K) * 500  # 700 bits per UE
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
        
        # Also calculate exact FBL error probabilities
        rate = data_size / blocklength
        err_prob_exact = np.ones_like(snr)
        valid_snr = snr > 0
        if np.any(valid_snr):
            err_prob_exact[valid_snr] = error_prob_fbl(snr[valid_snr], blocklength[valid_snr], rate[valid_snr])
        
        # Calculate objective value for the approximated model
        obj_val_approx = np.sum(weights * err_prob_app) - np.sum(later_weights)
        obj_val_exact = np.sum(weights * err_prob_exact) - np.sum(later_weights)
        
        print(f"\nPower allocation {i+1}: {power}")
        print(f"SNR values: {snr}")
        print(f"SNR (dB): {10*np.log10(snr)}")
        print(f"Error probabilities (approx FBL): {err_prob_app}")
        print(f"Error probabilities (exact FBL): {err_prob_exact}")
        print(f"Objective value (approx FBL): {obj_val_approx:.6f}")
        print(f"Objective value (exact FBL): {obj_val_exact:.6f}")
    
    # Solve the optimization problem using approximated FBL error model
    print("\n=== Solving Optimization Problem with Approximated FBL Error Model ===")
    try:
        P_opt, obj_value, status = power_allocation_fbl_approx(
            h_i, weights, later_weights, data_size, blocklength,
            N0, B, P_max, P_sum, theta, Tslot
        )
        
        # Print optimization results
        print("\n=== Optimization Results (Approximated FBL) ===")
        print(f"Optimization status: {status}")
        print(f"Objective function value: {obj_value:.6f}")  
        
        # Calculate SNR and error probabilities for verification
        D_i = m*N0*B/h_i
        snr_opt = P_opt/D_i
        
        # Calculate error probabilities (approximated FBL)
        err_prob_opt_approx = error_prob_fbl_approx(snr_opt, blocklength, data_size)
        mask = snr_opt <= 0
        err_prob_opt_approx[mask] = 1.0
        
        # Calculate error probabilities (exact FBL)
        rate = data_size / blocklength
        err_prob_opt_exact = np.ones_like(snr_opt)
        valid_snr = snr_opt > 0
        if np.any(valid_snr):
            err_prob_opt_exact[valid_snr] = error_prob_fbl(snr_opt[valid_snr], blocklength[valid_snr], rate[valid_snr])
        
        # Manually calculate the objective value
        obj_manual_approx = np.sum(weights * err_prob_opt_approx) - np.sum(later_weights)
        obj_manual_exact = np.sum(weights * err_prob_opt_exact) - np.sum(later_weights)
        
        print(f"Manually calculated objective (approx): {obj_manual_approx:.6f}")
        print(f"Manually calculated objective (exact): {obj_manual_exact:.6f}")
        
        print("\n=== Optimal Power Allocation (Approximated FBL) ===")
        print(f"{'UE Index':<10} {'Power (mW)':<15} {'SNR (dB)':<15} {'Error Prob (Approx)':<20} {'Error Prob (Exact)':<20}")
        print("-" * 80)
        
        # Display power allocation and related metrics for each UE
        total_power_allocated = 0
        for i, ue_idx in enumerate(user_indices):
            snr_db = 10 * np.log10(snr_opt[i]) if snr_opt[i] > 0 else float('-inf')
            error_prob_approx = err_prob_opt_approx[i]
            error_prob_exact = err_prob_opt_exact[i]
            total_power_allocated += P_opt[i]
            
            print(f"{ue_idx:<10} {P_opt[i]:<15.6f} {snr_db:<15.2f} {error_prob_approx:<20.6f} {error_prob_exact:<20.6f}")
            
        # Display the power allocation summary
        print("\n=== Power Allocation Summary ===")
        print(f"{'UE Index':<10} {'Power (mW)':<15} {'% of Total Power':<20}")
        print("-" * 45)
        
        for i, ue_idx in enumerate(user_indices):
            percentage = (P_opt[i] / total_power_allocated) * 100 if total_power_allocated > 0 else 0
            print(f"{ue_idx:<10} {P_opt[i]:<15.6f} {percentage:<20.2f}%")
        
        # Print SNR calculation details for clarity
        print("\n=== SNR Calculation Details ===")
        print("Formula used: SNR = Power / D_i, where D_i = m*N0*B/h_i")
        print("This expands to: SNR = Power*h_i / (m*N0*B)")
        print(f"{'UE Index':<10} {'Power (mW)':<15} {'Channel Gain':<15} {'D_i':<15} {'SNR Value':<15} {'SNR (dB)':<15}")
        print("-" * 85)
        
        for i, ue_idx in enumerate(user_indices):
            d_i_value = m*N0*B/h_i[i]
            snr_value = P_opt[i]/d_i_value if d_i_value > 0 else 0
            snr_db = 10 * np.log10(snr_value) if snr_value > 0 else float('-inf')
            
            # Calculate standard SNR for comparison (without m factor)
            std_snr = P_opt[i]*h_i[i]/(N0*B) if h_i[i] > 0 else 0
            std_snr_db = 10 * np.log10(std_snr) if std_snr > 0 else float('-inf')
            
            print(f"{ue_idx:<10} {P_opt[i]:<15.6f} {h_i[i]:<15.6e} {d_i_value:<15.6e} {snr_value:<15.6f} {snr_db:<15.2f}")
        
        print("\n=== Standard SNR Calculation (without m factor) ===")
        print("Formula: Standard SNR = Power*h_i / (N0*B)")
        print(f"{'UE Index':<10} {'Power (mW)':<15} {'Standard SNR':<15} {'Standard SNR (dB)':<20}")
        print("-" * 65)
        
        for i, ue_idx in enumerate(user_indices):
            std_snr = P_opt[i]*h_i[i]/(N0*B) if h_i[i] > 0 else 0
            std_snr_db = 10 * np.log10(std_snr) if std_snr > 0 else float('-inf')
            print(f"{ue_idx:<10} {P_opt[i]:<15.6f} {std_snr:<15.6f} {std_snr_db:<20.2f}")
            
        # Calculate total power
        total_power = np.sum(P_opt) + theta/Tslot * np.sum(data_size)
        print(f"\nTotal power consumption: {np.sum(P_opt):.6f} (transmit only)")
        print(f"Total power consumption: {total_power:.6f} / {P_sum:.6f} (including computation)")
        
        # Calculate success probabilities
        success_prob_approx = 1 - err_prob_opt_approx
        success_prob_exact = 1 - err_prob_opt_exact
        
        # Print successful transmission probabilities
        print("\n=== Successful Transmission Probabilities ===")
        print(f"Average success probability (approx): {np.mean(success_prob_approx):.6f}")
        print(f"Average success probability (exact): {np.mean(success_prob_exact):.6f}")
        
        # Simulate transmission success/failure like in wireless_fbl.py
        print("\n=== Transmission Success/Failure Simulation ===")
        np.random.seed(args.wireless_seed + 123)  # Use same approach as in wireless_fbl.py
        
        active_success_clients = []
        active_clients = [i for i in range(K) if P_opt[i] > 0]  # Only consider UEs with non-zero power
        fails = 0
        
        print(f"Total active clients: {len(active_clients)}")
        print(f"{'UE Index':<10} {'Success Prob':<15} {'Random Value':<15} {'Transmission':<15}")
        print("-" * 55)
        
        for idx in active_clients:
            # Generate random number between 0 and 1
            random_val = np.random.random()
            # Compare with success probability
            success = random_val < success_prob_approx[idx]
            
            if success:
                active_success_clients.append(idx)
                result = "SUCCESS"
            else:
                fails += 1
                result = "FAILURE"
                
            print(f"{idx:<10} {success_prob_approx[idx]:<15.6f} {random_val:<15.6f} {result:<15}")
        
        print(f"\nSuccessfully transmitted clients: {active_success_clients}")
        print(f"Number of failed transmissions: {fails} out of {len(active_clients)}")
        print(f"Simulated success rate: {len(active_success_clients)/len(active_clients) if len(active_clients) > 0 else 0:.4f}")
        print(f"Expected success rate: {np.mean(success_prob_approx[active_clients]) if len(active_clients) > 0 else 0:.4f}")
        
        # Show power vs error probability relationship
        print("\n=== Power vs Error Probability Relationship ===")
        print(f"{'UE Index':<10} {'Power (mW)':<15} {'Channel Gain':<15} {'D_i':<15} {'SNR':<15} {'SNR (dB)':<15} {'Error Prob':<15} {'Success Prob':<15}")
        print("-" * 110)
        
        # Sort by power allocation for better visualization
        sorted_indices = np.argsort(-P_opt)  # Sort by descending power
        for i in sorted_indices:
            ue_idx = user_indices[i]
            d_i_value = m*N0*B/h_i[i]
            snr_value = P_opt[i]/d_i_value
            snr_db = 10 * np.log10(snr_value) if snr_value > 0 else float('-inf')
            print(f"{ue_idx:<10} {P_opt[i]:<15.6f} {h_i[i]:<15.6e} {d_i_value:<15.6e} {snr_value:<15.6f} {snr_db:<15.2f} {err_prob_opt_approx[i]:<15.6f} {success_prob_approx[i]:<15.6f}")
        
    except Exception as e:
        print(f"Error in approximated FBL optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_fbl_optimization()