"""
modal_v300_benchmark.py
========================
Parallel V300 Capacity Scaling Benchmark on Modal GPU Cluster.

Usage:
  modal run scratch/modal_v300_benchmark.py
"""

import os
import sys
import time
import json
import modal

app = modal.App("v300-capacity-scaling")

SCRATCH_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRATCH_DIR)

# Container image with PyTorch CUDA support & mounted scratch code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2.0", "numpy")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(SCRATCH_DIR, remote_path="/root/attention-neuron/scratch")
)

@app.function(
    image=image,
    gpu="L4",  # Using NVIDIA L4 (24 GB VRAM) for peak memory stability
    timeout=3600  # Extended timeout for heavy sweeps (d_k=128, L=2048)
)
def run_single_worker(d_k: int, num_pairs: int):
    import torch
    os.chdir("/root/attention-neuron")
    sys.path.insert(0, "/root/attention-neuron")

    import scratch.prototype_v300_capacity_scaling as v300
    v300.CFG["device"] = "cuda"
    v300.device = torch.device("cuda")

    print(f"--> [Worker GPU {torch.cuda.get_device_name(0)}] Starting d_k={d_k}, num_pairs={num_pairs}")
    start_t = time.time()
    dk_res, np_res, comb_results = v300.run_single_combination(d_k, num_pairs)
    elapsed = time.time() - start_t
    print(f"<-- [Worker GPU] Completed d_k={d_k}, num_pairs={num_pairs} in {elapsed:.2f}s")
    return d_k, num_pairs, comb_results, elapsed

@app.local_entrypoint()
def main():
    print("=" * 80)
    print("LAUNCHING V300 PARALLEL SWEEP ACROSS MODAL GPU CLUSTER")
    print("=" * 80)

    d_k_list = [32, 64, 128]
    num_pairs_list = [32, 64, 128, 256]
    tasks = [(dk, np) for dk in d_k_list for np in num_pairs_list]

    print(f"Spawning {len(tasks)} parallel GPU workers on Modal...")
    start_t = time.time()

    results_matrix = {dk: {} for dk in d_k_list}

    for dk, np, comb_results, worker_t in run_single_worker.starmap(tasks):
        for key_name, data in comb_results.items():
            if key_name not in results_matrix[dk]:
                results_matrix[dk][key_name] = {}
            results_matrix[dk][key_name][np] = data
        print(f"  ✓ Received completed results for d_k={dk}, num_pairs={np} (took {worker_t:.2f}s)")

    total_elapsed = time.time() - start_t

    print("\n" + "=" * 80)
    print(f"PARALLEL MODAL GPU BENCHMARK COMPLETE")
    print(f"Total Wall-Clock Time: {total_elapsed:.2f}s ({total_elapsed/60.0:.2f} min)")
    print("=" * 80)

    import scratch.prototype_v300_capacity_scaling as v300
    full_output = {
        "config": v300.CFG,
        "results": results_matrix
    }

    local_output_dir = os.path.join(PROJECT_DIR, "results", "raw")
    os.makedirs(local_output_dir, exist_ok=True)
    local_output_file = os.path.join(local_output_dir, "v300_capacity_scaling.json")
    with open(local_output_file, "w") as f:
        json.dump(full_output, f, indent=2)

    print(f"Saved aggregated results to: {local_output_file}")

if __name__ == "__main__":
    main()
