from typing import Callable

import numpy as np
import torch
import iisignature
from joblib import Parallel, delayed


def calculate_signatures(paths: torch.Tensor, truncation_level: int) -> torch.Tensor:
    """
    Calculates the signatures of a batch of paths.

    Args:
        paths (torch.Tensor): A 3D tensor of shape (num_paths, num_steps, num_dimensions).
        trunctation_level (int): The signature truncation level.

    Returns:
        torch.Tensor: A 2D tensor of shape (num_paths, signature_length).
    """
    # The document specifies using iisignature directly for performance.
    # We must convert the PyTorch tensor to a NumPy array for iisignature.
    # The output is then converted back to a PyTorch tensor to stay in the ecosystem.

    # Detach and move tensor to CPU for numpy conversion
    paths_np = paths.detach().cpu().numpy()

    # Calculate signatures using iisignature
    signatures_np = iisignature.sig(paths_np, truncation_level)

    # Convert back to a PyTorch tensor
    signatures_torch = torch.from_numpy(signatures_np).type(dtype=torch.float32)

    # Return the tensor, ensuring it is on the correct device if needed.
    return signatures_torch.to(paths.device)


def _calculate_signature_batch(t_batch, paths, lookback, truncation_level, calculate_signatures_fn):
    """
    Calculates signatures for a whole batch of time steps `t`.
    This is the function that runs on each parallel core.
    """
    results_in_batch = []
    for t in t_batch:
        start_index = max(0, t - lookback)
        paths_slice = paths[:, start_index:t + 1, :]

        if paths_slice.shape[1] > 1:
            signature = calculate_signatures_fn(paths_slice, truncation_level)
            results_in_batch.append((t, signature))
        else:
            results_in_batch.append((t, None))

    return results_in_batch


def parallel_calculate_all_signatures_batched(
        paths: torch.Tensor,
        lookback: int,
        truncation_level: int,
        batch_size: int = 20  # A larger batch size is more efficient
) -> torch.Tensor:
    """
    Calculates signatures in parallel by dividing the work into efficient batches.
    """

    # Normalization (still needed here as it's part of the algorithm logic)
    num_features = paths.shape[2]

    num_steps = paths.shape[1]

    # 1. Divide the total range of time steps into batches.
    all_time_steps = range(num_steps - 1)
    num_batches = (len(all_time_steps) + batch_size - 1) // batch_size  # Ceiling division
    time_step_batches = np.array_split(all_time_steps, num_batches)

    # 2. Create a list of tasks, where each task processes one batch.
    tasks = [
        delayed(_calculate_signature_batch)(
            batch, paths, lookback, truncation_level, calculate_signatures
        ) for batch in time_step_batches
    ]

    # 3. Execute the batch tasks in parallel.
    # This will now show high CPU usage as the workers have substantial tasks.
    batch_results = Parallel(n_jobs=-1)(tasks)

    # 4. Assemble the results from all batches.
    # Flatten the list of lists into a single list of (t, signature) tuples.
    all_results = [item for sublist in batch_results for item in sublist]

    first_valid_result = next((sig for t, sig in all_results if sig is not None), None)
    if first_valid_result is None:
        return torch.empty(0)

    signature_dim = first_valid_result.shape[1]
    all_integrands_unscaled = torch.zeros(
        paths.shape[0], num_steps, signature_dim,
        dtype=first_valid_result.dtype, device=first_valid_result.device
    )

    for t, signature in all_results:
        if signature is not None:
            all_integrands_unscaled[:, t, :] = signature

    # Get the boolean mask for finite values
    # finite_mask = torch.isfinite(all_integrands_unscaled)

    # Use the mask to get a tensor with only finite values
    # finite_x = all_integrands_unscaled[finite_mask]

    # Find the maximum of the finite values and get the scalar
    # largest_finite_value = torch.max(finite_x).item()

    # all_integrands_unscaled = all_integrands_unscaled.clamp(min=0, max=largest_finite_value)
    # final_time_integrands = all_integrands_unscaled[:, -1, :]

    # mean = final_time_integrands.mean(dim=0, keepdim=True)
    # std = final_time_integrands.std(dim=0, keepdim=True) + 1e-8
    # all_integrands_scaled = (all_integrands_unscaled - mean.unsqueeze(1)) / std.unsqueeze(1)

    return all_integrands_unscaled
