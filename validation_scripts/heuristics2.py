import matplotlib.pyplot as plt
import numpy as np

from src.deepquant.heuristics import get_num_steps_v2

# --- Plotting Script ------------------------------------------------------
if __name__ == "__main__":
    # --- Configuration ---
    # Set your computational constraint
    MY_MAX_STEPS = 20000

    # Set your known data points
    X_0_2 = 10000  # N(0.2)
    X_0_5 = 2000  # N(0.5)

    # Points to plot
    known_H = [0.2, 0.5]
    known_N = [X_0_2, X_0_5]

    # --- Data Generation ---
    # Generate H values for a smooth curve
    # Start from 0.01 to show the asymptote without hitting H=0
    H_plot = np.linspace(0.01, 0.5, 400)  # Plot a bit past 0.5

    # Calculate N(H) for each H
    N_plot = [get_num_steps_v2(h, MY_MAX_STEPS, X_0_2, X_0_5) for h in H_plot]

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    # Plot the heuristic curve
    plt.plot(H_plot, N_plot, label="Heuristic N(H)", color="blue", zorder=2)

    # Plot the max_steps constraint line
    plt.axhline(y=MY_MAX_STEPS, color='red', linestyle='--',
                label=f"max_steps = {MY_MAX_STEPS}", zorder=1)

    # Plot the original data points
    plt.plot(known_H, known_N, 'ro', markersize=8,
             label="Known Data Points", zorder=3)

    # --- Formatting ---
    plt.title(f"Inverse Heuristic N(H) [N(0.2)={X_0_2}, N(0.5)={X_0_5}]")
    plt.xlabel("H value")
    plt.ylabel("Number of Steps (N)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set plot limits
    plt.ylim(0, MY_MAX_STEPS * 1.5)
    plt.xlim(0, 0.6)

    # Display the plot
    plt.show()