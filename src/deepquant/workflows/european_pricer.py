# deep-quant/src/deepquant_calibrator/core/pricing/european_pricer.py

import logging
import torch
import numpy as np
from abc import ABC, abstractmethod

# --- Import the SDE models for type hinting and usage ---
from ..models.sde import SDEModel, HestonModel, BergomiModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EuropeanPricerBase(ABC):
    """Abstract base class for all European option pricers."""

    @abstractmethod
    def price(self, sde_model: SDEModel, T: float, K: float, option_type: str) -> float:
        """Calculates the price of a European option for a given SDE model."""
        pass


class HestonEuropeanPricer(EuropeanPricerBase):
    """
    Prices European options under the Heston model using a fast, semi-analytical
    Fourier inversion method (e.g., Carr-Madan formula).
    """

    def _char_func(self, u, T, r, v0, kappa, theta, xi, rho):
        """ The characteristic function of log(S_T) for the Heston model. """
        d = torch.sqrt((kappa - 1j * rho * xi * u) ** 2 + (u ** 2 + 1j * u) * xi ** 2)
        g = (kappa - 1j * rho * xi * u - d) / (kappa - 1j * rho * xi * u + d)

        C = 1j * u * r * T + (kappa * theta / xi ** 2) * (
                (kappa - 1j * rho * xi * u - d) * T - 2 * torch.log((1 - g * torch.exp(-d * T)) / (1 - g))
        )
        B = (kappa - 1j * rho * xi * u - d) / xi ** 2 * ((1 - torch.exp(-d * T)) / (1 - g * torch.exp(-d * T)))

        return torch.exp(C + B * v0)

    def price(self, sde_model: HestonModel, T: float, K: float, option_type: str = 'put') -> float:
        """
        Calculates the Heston European option price using the proper P1, P2 formulation.
        """
        logging.info("Calculating Heston European price via Fourier method...")

        s0, v0, kappa, theta, xi, rho, r = (
            sde_model.s0, sde_model.v0, sde_model.kappa, sde_model.theta, sde_model.xi, sde_model.rho, sde_model.r
        )

        # Integration parameters
        integration_limit = 200.0
        num_points = 2000
        u = torch.linspace(1e-8, integration_limit, num_points)
        k = torch.log(torch.tensor(K))

        # Characteristic function of log(S_T) is phi(u) = E[exp(i*u*log(S_T))]
        # phi(u) = exp(i*u*log(s0)) * char_func_of_remainder
        # where char_func_of_remainder is our _char_func
        phi = torch.exp(1j * u * torch.log(torch.tensor(s0))) * self._char_func(u, T, r, v0, kappa, theta, xi, rho)
        phi_minus_i = torch.exp(1j * (u - 1j) * torch.log(torch.tensor(s0))) * self._char_func(u - 1j, T, r, v0, kappa,
                                                                                               theta, xi, rho)

        # Integrands for P1 and P2
        integrand1 = torch.real(torch.exp(-1j * u * k) * phi_minus_i / (1j * u * s0 * torch.exp(torch.tensor(r * T))))
        integrand2 = torch.real(torch.exp(-1j * u * k) * phi / (1j * u))

        P1 = 0.5 + (1.0 / np.pi) * torch.trapz(integrand1, u)
        P2 = 0.5 + (1.0 / np.pi) * torch.trapz(integrand2, u)

        # Final call price formula
        call_price = s0 * P1 - K * torch.exp(torch.tensor(-r * T)) * P2

        if option_type == 'call':
            return call_price.item()
        else:  # Calculate put price via put-call parity
            put_price = call_price - s0 + K * np.exp(-r * T)
            return put_price.item()


class BergomiEuropeanPricer(EuropeanPricerBase):
    """
    Prices European options under the Bergomi model using a dedicated,
    high-precision Monte Carlo simulation.
    """

    def price(self, sde_model: BergomiModel, T: float, K: float, option_type: str = 'put') -> float:
        logging.info("Calculating high-precision Bergomi European price for control variate...")

        # Use a large number of paths for a single, accurate estimate.
        num_control_paths = 200000
        # Use a high number of steps for accuracy.
        num_steps = int(T * 252)

        paths, _ = sde_model.simulate_paths_qmc_antithetic(num_control_paths, num_steps, T)

        S_T = paths[:, -1, 0]
        payoffs = torch.clamp(K - S_T, min=0) if option_type == 'put' else torch.clamp(S_T - K, min=0)
        price = torch.mean(payoffs) * np.exp(-sde_model.r * T)

        return price.item()