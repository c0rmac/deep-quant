import unittest
import torch

from src.deepquant.models.sde import HestonModel, BergomiModel, SDEFactory


class TestSDESimulator(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for SDE models."""
        self.num_paths = 100
        self.num_steps = 50
        self.T = 1.0
        self.s0 = 100.0
        self.v0 = 0.04
        self.r = 0.05

        # Common parameters for Heston
        self.heston_params = {
            's0': self.s0, 'v0': self.v0, 'kappa': 2.0,
            'theta': 0.04, 'xi': 0.3, 'rho': -0.7, 'r': self.r
        }

        # Common parameters for Bergomi
        self.bergomi_params = {
            's0': self.s0, 'v0': self.v0, 'eta': 1.9, 'rho': -0.9, 'r': self.r
        }

    def test_heston_model_simulation(self):
        """
        Test the Heston model simulation for output shapes and basic properties.
        """
        print("\n--- Testing Heston Model Simulation ---")
        heston_model = HestonModel(**self.heston_params)

        # The method now returns paths and Brownian increments dW
        paths, dW = heston_model.simulate_paths(self.num_paths, self.num_steps, self.T)

        # 1. Check output shapes
        self.assertEqual(paths.shape, (self.num_paths, self.num_steps + 1, 2))
        self.assertEqual(dW.shape, (self.num_paths, self.num_steps))
        print("Heston path and dW shapes are correct.")

        # 2. Check for non-negative volatility
        volatility_paths = paths[:, :, 1]
        self.assertTrue(torch.all(volatility_paths >= 0), "Volatility should be non-negative.")
        print("Volatility is non-negative.")

        # 3. Check expected stock price at maturity (a loose check)
        expected_mean = self.s0 * torch.exp(torch.tensor(self.r * self.T))
        actual_mean = torch.mean(paths[:, -1, 0])
        tolerance = 5.0
        self.assertLess(torch.abs(expected_mean - actual_mean), tolerance,
                        f"Mean of S_T ({actual_mean:.2f}) is too far from expected value ({expected_mean:.2f}).")
        print(f"Mean of final stock price is plausible (Actual: {actual_mean:.2f}, Expected: {expected_mean:.2f}).")
        print("--- Heston Model Test Passed ---")

    def test_bergomi_model_simulation(self):
        """
        Test the concrete Bergomi model simulation for output shapes.
        """
        print("\n--- Testing Bergomi Model Simulation ---")
        # Use a valid Hurst parameter H < 0.5
        H = 0.1
        bergomi_model = BergomiModel(H=H, **self.bergomi_params)

        # The method now returns paths and Brownian increments dB (aliased as dW)
        paths, dW = bergomi_model.simulate_paths(self.num_paths, self.num_steps, self.T)

        # 1. Check output shapes
        self.assertEqual(paths.shape, (self.num_paths, self.num_steps + 1, 2))
        self.assertEqual(dW.shape, (self.num_paths, self.num_steps))
        print("Bergomi path and dW shapes are correct.")

        # 2. Check for non-negative volatility
        volatility_paths = paths[:, :, 1]
        self.assertTrue(torch.all(volatility_paths >= 0), "Volatility should be non-negative.")
        print("Volatility is non-negative.")
        print("--- Bergomi Model Test Passed ---")

    def test_sde_factory(self):
        """
        Test the SDEFactory to ensure it creates the correct model type.
        """
        print("\n--- Testing SDE Factory ---")
        factory = SDEFactory()

        # 1. Test Heston model creation (H=0.5)
        heston_model = factory.create_model(H=0.5, **self.heston_params)
        self.assertIsInstance(heston_model, HestonModel)
        print("Factory correctly created HestonModel for H=0.5.")

        # 2. Test Bergomi model creation (H<0.5)
        bergomi_model = factory.create_model(H=0.1, **self.bergomi_params)
        self.assertIsInstance(bergomi_model, BergomiModel)
        print("Factory correctly created BergomiModel for H=0.1.")

        # 3. Test for unsupported H value (edge case)
        with self.assertRaises(ValueError):
            factory.create_model(H=0.0, **self.bergomi_params)
        print("Factory correctly raised ValueError for unsupported H=0.0.")
        print("--- SDE Factory Test Passed ---")


if __name__ == '__main__':
    unittest.main()