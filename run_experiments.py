import subprocess
import time

commands = [
    # MNIST experiments with total_blocklength=1000
    "python main_fed_fbl.py --dataset mnist --round 200 --iid iid --selection \"weighted_random\" --scenario PER --total_blocklength 1000 --packet_size 100",
    "python main_fed_fbl.py --dataset mnist --round 200 --iid niid --selection \"weighted_random\" --scenario PER --total_blocklength 1000 --packet_size 100",
    
    # CIFAR experiments with total_blocklength=1000
    "python main_fed_fbl.py --dataset cifar --round 800 --iid iid --selection \"weighted_random\" --scenario PER --total_blocklength 1000 --packet_size 100",
    "python main_fed_fbl.py --dataset cifar --round 800 --iid niid --selection \"weighted_random\" --scenario PER --total_blocklength 1000 --packet_size 100",
    
    # MNIST experiments with total_blocklength=3000
    "python main_fed_fbl.py --dataset mnist --round 200 --iid iid --selection \"weighted_random\" --scenario PER --total_blocklength 3000 --packet_size 100",
    "python main_fed_fbl.py --dataset mnist --round 200 --iid niid --selection \"weighted_random\" --scenario PER --total_blocklength 3000 --packet_size 100",
    
    # CIFAR experiments with total_blocklength=3000
    "python main_fed_fbl.py --dataset cifar --round 800 --iid iid --selection \"weighted_random\" --scenario PER --total_blocklength 3000 --packet_size 100",
    "python main_fed_fbl.py --dataset cifar --round 800 --iid niid --selection \"weighted_random\" --scenario PER --total_blocklength 3000 --packet_size 100"
]

for i, cmd in enumerate(commands):
    print(f"\n[{i+1}/{len(commands)}] Running: {cmd}\n")
    start_time = time.time()
    
    # Execute the command
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    
    elapsed_time = time.time() - start_time
    print(f"\nCommand completed in {elapsed_time:.2f} seconds")
    print("-" * 80)