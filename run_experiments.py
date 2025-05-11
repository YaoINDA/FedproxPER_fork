import subprocess
import time
from datetime import datetime

# Define the commands to run
commands = [
    # # MNIST with linear FBL selection (blocklength 800)
    # "python main_fed_fbl_linear.py --dataset mnist --round 500 --iid iid --selection \"linear_fbl\" --scenario PER --total_blocklength 800 --packet_size 100 --name blocklength_var_mnist",
    
    # # MNIST with linear FBL selection (blocklength 1000)
    # "python main_fed_fbl_linear.py --dataset mnist --round 500 --iid iid --selection \"linear_fbl\" --scenario PER --total_blocklength 1000 --packet_size 100 --name blocklength_var_mnist",
    
    # # MNIST with linear FBL selection (blocklength 1200)
    # "python main_fed_fbl_linear.py --dataset mnist --round 500 --iid iid --selection \"linear_fbl\" --scenario PER --total_blocklength 1200 --packet_size 100 --name blocklength_var_mnist",
    
    # # MNIST with linear FBL selection (blocklength 1400)
    # "python main_fed_fbl_linear.py --dataset mnist --round 500 --iid iid --selection \"linear_fbl\" --scenario PER --total_blocklength 1400 --packet_size 100 --name blocklength_var_mnist",
    
    # # MNIST with linear FBL selection (blocklength 1600)
    # "python main_fed_fbl_linear.py --dataset mnist --round 500 --iid iid --selection \"linear_fbl\" --scenario PER --total_blocklength 1600 --packet_size 100 --name blocklength_var_mnist",
    
    # # MNIST with linear FBL selection (blocklength 1800)
    # "python main_fed_fbl_linear.py --dataset mnist --round 500 --iid iid --selection \"linear_fbl\" --scenario PER --total_blocklength 1800 --packet_size 100 --name blocklength_var_mnist",
    
    # # MNIST with linear FBL selection (blocklength 2000)
    # "python main_fed_fbl_linear.py --dataset mnist --round 500 --iid iid --selection \"linear_fbl\" --scenario PER --total_blocklength 2000 --packet_size 100 --name blocklength_var_mnist",


    # MNIST with linear FBL selection (blocklength 1000)
    "python main_fed_fbl_linear.py --dataset mnist --round 500 --iid niid --selection \"linear_fbl\" --scenario PER --total_blocklength 1000 --packet_size 100 --name blocklength_1000_mnist",

    # MNIST with linear FBL selection (blocklength 1010)
    "python main_fed_fbl_linear.py --dataset mnist --round 500 --iid niid --selection \"linear_fbl\" --scenario PER --total_blocklength 1010 --packet_size 100 --name blocklength_1000_mnist",
     
    # MNIST with weighted random selection (blocklength 1000)
    "python main_fed_fbl.py --dataset mnist --round 500 --iid niid --selection \"weighted_random\" --scenario PER --total_blocklength 1000 --packet_size 100 --name blocklength_1000_mnist",
    
    # MNIST with best channel ratio selection (blocklength 1000)
    "python main_fed_fbl.py --dataset mnist --round 500 --iid niid --selection \"best_channel\" --scenario PER --total_blocklength 1000 --packet_size 100 --name blocklength_1000_mnist",
    
    # MNIST with solve_opti_loss_size2 selection as benchmark (blocklength 1000)
    "python main_fed_benchmark.py --dataset mnist --round 500 --iid niid --selection \"solve_opti_loss_size2\" --scenario PER --total_blocklength 1000 --packet_size 100 --name blocklength_1000_mnist",
    
    # CIFAR with linear FBL selection (blocklength 1000)
    "python main_fed_fbl_linear.py --dataset cifar --round 1000 --iid niid --selection \"linear_fbl\" --scenario PER --total_blocklength 1000 --packet_size 100 --name blocklength_1000_cifar",

    # CIFAR with linear FBL selection (blocklength 1010)
    "python main_fed_fbl_linear.py --dataset cifar --round 1000 --iid niid --selection \"linear_fbl\" --scenario PER --total_blocklength 1010 --packet_size 100 --name blocklength_1000_cifar",
     
    # CIFAR with weighted random selection (blocklength 1000)
    "python main_fed_fbl.py --dataset cifar --round 1000 --iid niid --selection \"weighted_random\" --scenario PER --total_blocklength 1000 --packet_size 100 --name blocklength_1000_cifar",
    
    # CIFAR with best channel ratio selection (blocklength 1000)
    "python main_fed_fbl.py --dataset cifar --round 1000 --iid niid --selection \"best_channel\" --scenario PER --total_blocklength 1000 --packet_size 100 --name blocklength_1000_cifar",
    
    # CIFAR with solve_opti_loss_size2 selection as benchmark (blocklength 1000)
    "python main_fed_benchmark.py --dataset cifar --round 1000 --iid niid --selection \"solve_opti_loss_size2\" --scenario PER --total_blocklength 1000 --packet_size 100 --name blocklength_1000_cifar",
    
    # # MNIST with linear FBL selection (blocklength 2000)
    # "python main_fed_fbl_linear.py --dataset mnist --round 500 --iid iid --selection \"linear_fbl\" --scenario PER --total_blocklength 2000 --packet_size 100 --name blocklength_2000_mnist",
    
    # # MNIST with weighted random selection (blocklength 2000)
    # "python main_fed_fbl.py --dataset mnist --round 500 --iid iid --selection \"weighted_random\" --scenario PER --total_blocklength 2000 --packet_size 100 --name blocklength_2000_mnist",
    
    # # MNIST with best channel ratio selection (blocklength 2000)
    # "python main_fed_fbl.py --dataset mnist --round 500 --iid iid --selection \"best_channel\" --scenario PER --total_blocklength 2000 --packet_size 100 --name blocklength_2000_mnist",
    
    # # MNIST with solve_opti_loss_size2 selection as benchmark (blocklength 2000)
    # "python main_fed_benchmark.py --dataset mnist --round 500 --iid iid --selection \"solve_opti_loss_size2\" --scenario PER --total_blocklength 2000 --packet_size 100 --name blocklength_2000_mnist",

]

# Create a log file for the experimental results
log_filename = f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

with open(log_filename, "w") as log_file:
    log_file.write(f"Federated Learning Experiments Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 80 + "\n\n")
    
    # Run each command sequentially
    for i, cmd in enumerate(commands):
        start_time = time.time()
        
        # Print and log the command being executed
        print(f"\n[{i+1}/{len(commands)}] Running: {cmd}\n")
        log_file.write(f"Experiment {i+1}/{len(commands)}\n")
        log_file.write(f"Command: {cmd}\n")
        log_file.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.flush()  # Ensure the log is written even if the program crashes
        
        try:
            # Execute the command
            process = subprocess.Popen(cmd, shell=True)
            process.wait()
            
            # Calculate and log the elapsed time
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            print(f"\nCommand completed in {time_str}")
            log_file.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Duration: {time_str}\n")
            log_file.write(f"Exit code: {process.returncode}\n")
            log_file.write("-" * 80 + "\n\n")
            log_file.flush()
            
        except Exception as e:
            print(f"\nError executing command: {e}")
            log_file.write(f"Error: {e}\n")
            log_file.write("-" * 80 + "\n\n")
            log_file.flush()

print(f"\nAll experiments completed. Log saved to {log_filename}")