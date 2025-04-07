import numpy as np
from selections.wireless import dB2power, wireless_param, update_wireless
from utils.options import args_parser

if __name__ == '__main__':
    # Parse arguments
    args = args_parser()
    
    # Initialize dummy data weights for wireless param initialization
    data_weight = np.ones(args.total_UE) / args.total_UE
    nb_data_assigned = args.total_UE  # Dummy value
    
    # Initialize wireless parameters
    wireless_arg = wireless_param(args, data_weight, nb_data_assigned)
    
    # Update wireless parameters with random seed
    wireless_arg = update_wireless(args, wireless_arg, args.wireless_seed)
    
    # Get PER from success probability
    per = 1 - wireless_arg['success prob']
    
    # Print results
    print("\nPacket Error Rate (PER) for each UE:")
    for i in range(args.total_UE):
        print(f"UE {i}: {per[i]:.4f}")
    
    print(f"\nAverage PER across all UEs: {np.mean(per):.4f}")
    print(f"Min PER: {np.min(per):.4f}")
    print(f"Max PER: {np.max(per):.4f}")
