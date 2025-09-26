import torch
import iisignature


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