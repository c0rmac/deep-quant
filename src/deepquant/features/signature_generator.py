import heapq

import torch
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
import collections


# This function must be at the top level for the ProcessPoolExecutor to work.
def calculate_signatures_static(paths_slice, truncation_level):
    """A static version of the signature calculation for parallel execution."""
    # This import happens in the worker process.
    from ..features.signature_calculator import calculate_signatures
    return calculate_signatures(paths_slice, truncation_level)


class BatchSignatureGenerator:
    """
    A generator that computes signatures in parallel batches.
    """

    def __init__(self, paths, truncation_level, batch_size=4):
        self.paths = paths
        self.truncation_level = truncation_level
        self.batch_size = batch_size
        self.executor = ProcessPoolExecutor()

        num_steps = paths.shape[1]
        self.time_steps_to_process = collections.deque(range(num_steps - 2, -1, -1))

        # This deque will be our buffer for completed signatures
        self.results_buffer = collections.deque()

    def __enter__(self):
        # We can compute the first batch at startup
        self._compute_next_batch()
        return self

    def _compute_next_batch(self):
        """Submits a new batch of tasks and waits for them to complete."""
        tasks_in_batch = []
        # Gather tasks for the next batch
        for _ in range(self.batch_size):
            if not self.time_steps_to_process: break
            t = self.time_steps_to_process.popleft()
            paths_slice = self.paths[:, max(0, t - 500):t + 1, :]
            # paths_slice = self.paths[:, :t + 1, :]
            future = self.executor.submit(
                calculate_signatures_static, paths_slice, self.truncation_level
            )
            tasks_in_batch.append((t, future))

        if not tasks_in_batch:
            return

        # Wait for the entire batch to complete
        futures_only = [future for _, future in tasks_in_batch]
        wait(futures_only, return_when=ALL_COMPLETED)

        # Store the results in a temporary dictionary to sort them
        batch_results = {}
        for t, future in tasks_in_batch:
            batch_results[t] = future.result()

        # Add the sorted results to our buffer (in correct chronological order)
        for t in sorted(batch_results.keys(), reverse=True):
            self.results_buffer.append(batch_results[t])

    def __iter__(self):
        return self

    def __next__(self):
        # If the buffer is empty, compute the next batch.
        if not self.results_buffer:
            if not self.time_steps_to_process:
                raise StopIteration
            print(f"Signature buffer empty. Computing next batch of {self.batch_size}...")
            self._compute_next_batch()

        # If still no results, it means we are truly done.
        if not self.results_buffer:
            raise StopIteration

        return self.results_buffer.popleft()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown()