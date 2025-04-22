import numpy as np
from scipy.stats import norm
from selections.function_power_allocation_FBL_approx import power_allocation_fbl_approx, error_prob_fbl_approx

def debug_fbl_optimization_with_custom_h_i():
    # Set parameters matching your observed values
    total_UE = 10
    
    # Initialize wireless parameters
    wireless_arg = {
        'N0': 1e-15,  # Noise power spectral density
        'B': 1e6,     # Bandwidth
        'm': 0.023,   # Fading parameter
        'P_max': 10,  # Maximum power per UE (mW)
        'P_sum': 60,  # Total power budget (mW)
        'theta': 1e-8,  # Energy coefficient
        'Tslot': 1.3,   # Time slot duration
    }
    
    # Your observed h_i values
    custom_h_i = np.array([
        1.44490553e-06, 4.99449143e-08, 1.25188703e-08, 1.04636041e-08,
        2.03194699e-09, 5.25738706e-10, 2.07041340e-09, 9.48490948e-10,
        2.30571631e-09, 2.25629971e-09
    ])
    
    wireless_arg['h_i'] = custom_h_i
    
    # Use the same blocklength and packet size as seen in your output
    blocklength = np.ones(total_UE) * 100
    packet_size = np.ones(total_UE) * 100
    weights = np.ones(total_UE)
    later_weights = np.zeros(total_UE)
    
    # Extract parameters
    h_i = wireless_arg['h_i']
    N0 = wireless_arg['N0']
    B = wireless_arg['B']
    P_max = wireless_arg['P_max']
    P_sum = wireless_arg['P_sum']
    theta = wireless_arg['theta']
    Tslot = wireless_arg['Tslot']
    
    print("=== Parameters ===")
    print(f"blocklength: {blocklength}")
    print(f"packet_size: {packet_size}")
    print(f"h_i: {h_i}")
    
    # Call the exact function used in wireless_FBL.py
    try:
        print("\n=== Calling power_allocation_fbl_approx ===")
        P_opt, obj_val, status = power_allocation_fbl_approx(
            h_i, weights, later_weights, packet_size, blocklength,
            N0, B, P_max, P_sum, theta, Tslot
        )
        
        print(f"Status: {status}")
        print(f"Objective value: {obj_val}")
        print(f"Power allocation: {P_opt}")
        print(f"weights:{weights}")
        print(f"later_weights:{later_weights}")
        # Calculate error probabilities if optimization succeeded
        if P_opt is not None:
            # Calculate SNR for each user
            snr = P_opt * h_i / (N0 * B)
            
            # Calculate error probabilities using the same function as in wireless_FBL.py
            err_prob = error_prob_fbl_approx(snr, blocklength, packet_size)
            
            print("\n=== Error Probabilities ===")
            print(f"{'UE':<4} {'Power (mW)':<12} {'SNR (dB)':<12} {'Error Prob':<12}")
            print("-" * 40)
            for i in range(total_UE):
                snr_db = 10 * np.log10(snr[i]) if snr[i] > 0 else float('-inf')
                print(f"{i:<4} {P_opt[i]:<12.6f} {snr_db:<12.2f} {err_prob[i]:<12.6f}")
            print(f"Average error probability: {np.mean(err_prob):.6f}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_fbl_optimization_with_custom_h_i()