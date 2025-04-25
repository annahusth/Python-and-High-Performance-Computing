from os.path import join
import sys

import numpy as np
from numba import cuda
import time

##### LOAD DATA FUNCTION #####
def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

##### CUDA KERNEL: Performs one Jacobi iteration #####
@cuda.jit
def jacobi_kernel(u, u_new, mask):
    i, j = cuda.grid(2) # Get the 2D index of this thread in the grid

    #Consider only interor points (non-edges)
    if 1 <= i <= u.shape[0]-1 and 1 <= j < u.shape[1]-1: 
        if mask[i-1, j-1]: # Check if this is a real interior point (1 = yes, 0 = no) - shifted because mask is smaller (no border)
            u_new[i,j] = 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])


##### HELPER FUNCTION #####
def jacobi_cuda(u_host, interior_mask, max_iter):

    #Transfer arrays to device
    u_device = cuda.to_device(u_host)
    u_new_device = cuda.device_array_like(u_device)

    mask_device = cuda.to_device(interior_mask)

    #Thread block and grid dimensions
    threads_per_block = (16,16)
    blocks_per_grid_x = (u_host.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (u_host.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    for _ in range(max_iter):

        #Launching and calling CUDA kernel
        jacobi_kernel[blocks_per_grid, threads_per_block](u_device, u_new_device, mask_device)

        #Updating temperature and preparing for next iteration (swapping u and u_new for next interation)
        u_device, u_new_device = u_new_device, u_device 

    #Store result in u_device
    return u_device.copy_to_host()


##### STATISTICS #####
def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


if __name__ == '__main__':
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run CUDA jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)
    run_times = []
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        start = time.time() #run time start
        u = jacobi_cuda(u0, interior_mask, MAX_ITER) #using CUDA kernel
        end = time.time()
        elapsed = end - start
        run_times.append(elapsed) #list of all run times
        all_u[i] = u
        print(f"\n Run time using CUDA kernel for {building_ids[i]}: {elapsed:.4f} seconds")

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))


##### AVERAGE RUN TIME #####
avg_run_time = sum(run_times) / len(run_times)
print(f"\n Average run time using CUDA kernel: {avg_run_time:.2f} seconds")

