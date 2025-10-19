from typing import Tuple, List

import torch
import numpy as np
from joblib import Parallel, delayed

from .sde import SDEModel


def _generate_one_batch(
        sde: SDEModel,
        batch_num,
        num_paths_in_batch,
        num_steps,
        T
):
    """
    Helper function executed by each parallel worker to generate one batch of paths.
    """
    # Ensure each parallel job gets a unique, reproducible random seed

    # --- IMPORTANT ---
    # Make sure your simulation function can accept and use a seed.
    # If it doesn't, you might need to set torch.manual_seed(run_seed) here.
    paths, dW = sde.simulate_paths_qmc_antithetic(
        num_paths=num_paths_in_batch,
        num_steps=num_steps,
        T=T,
    )
    return paths, dW

def generate_paths_in_batches_individual(
    sde: SDEModel,
    num_paths: int,
    num_steps: int,
    T: float,
    num_batches: int,
    base_seed: int = 42
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generates multiple sets of QMC paths in parallel and returns each set individually.

    Args:
        num_paths (int): The total number of paths to generate across all batches.
        num_steps (int): The number of time steps per path.
        T (float): The time to maturity.
        num_batches (int): The number of parallel jobs/batches to create.
        base_seed (int): A seed for reproducibility.

    Returns:
        A list of tuples. Each tuple contains the results from one batch:
        [(paths_batch_1, dW_1, ...), (paths_batch_2, dW_2, ...), ...]
    """
    if num_paths % num_batches != 0:
        print(f"Warning: num_paths ({num_paths}) is not perfectly divisible by num_batches ({num_batches}).")

    paths_per_batch = num_paths
    paths_distribution = [paths_per_batch] * num_batches

    # --- Run the batch generation in parallel ---
    results_list = Parallel(n_jobs=-1)(
        delayed(_generate_one_batch)(sde, i, n_paths, num_steps, T)
        for i, n_paths in enumerate(paths_distribution)
    )

    return results_list

def generate_paths_in_batches(
        sde: SDEModel,
        num_paths: int,
        num_steps: int,
        T: float,
        num_batches: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a large set of QMC paths by creating smaller batches in parallel.

    Args:
        num_paths (int): The total number of paths to generate.
        num_steps (int): The number of time steps per path.
        T (float): The time to maturity.
        num_batches (int): The number of parallel jobs to split the work into.
        base_seed (int): A seed for reproducibility.

    Returns:
        A tuple containing the three combined tensors:
        (combined_paths, combined_dW, combined_final_martingale_values)
    """
    if num_paths % num_batches != 0:
        print(f"Warning: num_paths ({num_paths}) is not perfectly divisible by num_batches ({num_batches}).")

    # Determine the number of paths for each batch
    paths_per_batch = num_paths // num_batches
    paths_distribution = [paths_per_batch] * num_batches
    # Distribute the remainder among the first few batches
    for i in range(num_paths % num_batches):
        paths_distribution[i] += 1

    print(f"Distributing {num_paths} paths into {num_batches} parallel batches...")

    # --- Run the batch generation in parallel ---
    # `n_jobs=-1` uses all available CPU cores.
    # `delayed` creates a lightweight task for each batch.
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_generate_one_batch)(sde, i, n_paths, num_steps, T)
        for i, n_paths in enumerate(paths_distribution)
    )

    # --- Unpack and combine the results ---
    # `results` is a list of tuples, e.g., [(paths1, dW1, ...), (paths2, dW2, ...)]
    all_paths = [res[0] for res in results]
    all_dW = [res[1] for res in results]

    # Concatenate the results from all batches along the 'paths' dimension (dim=0)
    combined_paths = torch.cat(all_paths, dim=0)
    combined_dW = torch.cat(all_dW, dim=0)

    print("\n--- Parallel path generation complete ---")
    print(f"Final combined paths shape: {combined_paths.shape}")

    return combined_paths, combined_dW
