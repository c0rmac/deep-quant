import numpy as np


def get_num_steps(H: float, max_steps: int, x_0_2: int, x_0_4: int, x_0_5: int) -> int:
    """
    Calculates the number of steps using a spatially blended heuristic.

    This function is designed to satisfy multiple complex constraints by
    blending two different models across the domain H in (0, 0.5]:

    1.  Fits three data points:
        - N(0.2) = x_0_2
        - N(0.4) = x_0_4
        - N(0.5) = x_0_5
    2.  Satisfies the asymptotic constraint N(H) -> +inf as H -> 0.
    3.  Behaves like an inverse function in the region [0.2, 0.4].
    4.  Behaves like a linear function in the region [0.4, 0.5].

    ---
    Methodology:
    ---
    1.  **Inverse Model (N1)**: A function N1(H) = k/H + c is created.
        It is fit to the two data points (0.2, x_0_2) and (0.4, x_0_4).
        This model governs the behavior near H=0.

    2.  **Linear Model (NL)**: A function NL(H) = mH + c is created.
        It is fit to the two data points (0.4, x_0_4) and (0.5, x_0_5).
        This model governs the behavior near H=0.5.

    3.  **Blending Function (lambda(H))**: A linear blending function,
        lambda(H), is used to transition smoothly from N1 to NL.
        - At H=0.2, lambda(H) = 0 (output is 100% N1).
        - At H=0.5, lambda(H) = 1 (output is 100% NL).
        - At H=0.4, both models are exact, so the blend is seamless.

    4.  **Final Heuristic**:
        N(H) = (1 - lambda(H)) * N1(H) + lambda(H) * NL(H)

    Args:
        H (float): The input parameter, expected in the interval (0, 0.5].
        max_steps (int): The maximum allowable steps (computing constraint).
        x_0_2 (int): The known number of steps at H=0.2.
        x_0_4 (int): The known number of steps at H=0.4.
        x_0_5 (int): The known number of steps at H=0.5.

    Returns:
        int: The constrained, rounded number of steps for the given H.
    """

    # Handle the H=0 boundary case
    if H == 0:
        k_1 = 0.4 * (x_0_2 - x_0_4)
        return max_steps if k_1 > 0 else 1

    if H < 0:
        raise ValueError("H must be non-negative.")

    # --- 1. Define N1 (Inverse Model, fit to 0.2 and 0.4) ---
    k_1 = 0.4 * (x_0_2 - x_0_4)
    c_1 = (2.0 * x_0_4) - x_0_2
    N1_H = (k_1 / H) + c_1

    # If H is in the "pure N1" region, return early.
    if H <= 0.2:
        N_H_rounded = round(N1_H)
        if N_H_rounded < 1: N_H_rounded = 1
        return int(min(N_H_rounded, max_steps))

    # --- 2. Define NL (Linear Model, fit to 0.4 and 0.5) ---
    m_L = 10.0 * (x_0_5 - x_0_4)
    c_L = x_0_4 - m_L * 0.4
    NL_H = (m_L * H) + c_L

    # If H is in the "pure NL" region, return early.
    if H >= 0.5:
        N_H_rounded = round(NL_H)
        if N_H_rounded < 1: N_H_rounded = 1
        return int(min(N_H_rounded, max_steps))

    # --- 3. Define the smooth blending function for H in (0.2, 0.5) ---
    # Normalize H to t in [0, 1]
    t = (H - 0.2) / 0.3

    # Use the smoothstep function: S(t) = 3t^2 - 2t^3
    lambda_H = (3.0 * t ** 2) - (2.0 * t ** 3)

    # --- 4. Calculate the blended heuristic ---
    N_H_unbounded = (1.0 - lambda_H) * N1_H + lambda_H * NL_H

    # --- 5. Apply constraints ---
    N_H_rounded = round(N_H_unbounded)

    if N_H_rounded < 1:
        N_H_rounded = 1

    final_steps = int(min(N_H_rounded, max_steps))

    return final_steps


def get_num_steps_v2(H: float, max_steps: int, x_0_2: int, x_0_5: int) -> int:
    """
    Calculates num_steps using an inverse model (k/H + c)
    derived from N(0.2) = x_0_2 and N(0.5) = x_0_5.
    """

    # Handle the H=0 boundary case (N(H) -> inf)
    if H == 0:
        # We only go to infinity if the k term is positive
        k = (x_0_2 - x_0_5) / 3.0
        if k > 0:
            return max_steps
        else:
            # If k is negative, N(H) -> -inf, which is non-physical
            # We'll return 1 as the minimum step count.
            return 1

    if H < 0:
        raise ValueError("H must be non-negative.")

    # --- 1. Derive k and c ---
    # x_0_2 = k/0.2 + c  =>  x_0_2 = 5k + c
    # x_0_5 = k/0.5 + c  =>  x_0_5 = 2k + c
    # Subtracting: x_0_2 - x_0_5 = 3k

    k = (x_0_2 - x_0_5) / 3.0

    # c = x_0_5 - 2k
    c = x_0_5 - 2.0 * k

    # --- 2. Calculate N(H) ---
    N_H_unbounded = (k / H) + c

    # --- 3. Apply constraints ---

    # Round to nearest integer
    N_H_rounded = round(N_H_unbounded)

    # Ensure steps are at least 1 (in case c is large and negative)
    if N_H_rounded < 1:
        N_H_rounded = 1

    # Apply the maximum step constraint
    final_steps = int(min(N_H_rounded, max_steps))

    return final_steps