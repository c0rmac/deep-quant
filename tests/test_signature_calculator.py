import unittest
import torch

from src.deepquant.features.signature_calculator import calculate_signatures


class TestSignatureCalculator(unittest.TestCase):

    def setUp(self):
        """Set up a sample path tensor."""
        self.num_paths = 10
        self.num_steps = 20
        self.num_dims = 2  # e.g., (stock_price, volatility)
        self.paths = torch.randn(self.num_paths, self.num_steps, self.num_dims)

    def test_signature_dimension(self):
        """
        Test if the calculated signature has the correct length.
        """
        print("\n--- Testing Signature Calculator Dimension ---")
        for level in range(1, 5):
            with self.subTest(truncation_level=level):
                # Theoretical dimension of a signature for a d-dim path at level M
                # is (d^(M+1) - 1) / (d - 1).
                # The iisignature library omits the zero-th level term (the constant 1).
                full_dim = (self.num_dims ** (level + 1) - 1) // (self.num_dims - 1)
                expected_dim = full_dim - 1  # <-- CORRECTED LINE

                signatures = calculate_signatures(self.paths, level)

                self.assertEqual(signatures.shape[0], self.num_paths)
                self.assertEqual(signatures.shape[1], expected_dim)
                print(f"Level {level}: Correct signature dimension ({expected_dim}).")
        print("--- Signature Calculator Test Passed ---")


if __name__ == '__main__':
    unittest.main()