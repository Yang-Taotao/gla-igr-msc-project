"""
VI target distribution classes repo.
"""
# Library import
import jax.numpy as jnp
from data import gw_fim

# Dist - BVM distribution


class BivariateVonMises:
    """
    Original BVM class
    """
    def __init__(self, loc, concentration, correlation):
        self.data_mu, self.data_nu = loc
        self.data_k1, self.data_k2 = concentration
        self.data_k3 = correlation


    # Target dist, need to switch to working dist of fim density
    def log_prob(self, data_x):
        '''
        Get the log probability distribution

        Take in 2d param
        Feed 2d param to external func for some density result (dtype: float ish)
        The float is the return of log_prob()
        '''
        # 2-D parameters
        phi, psi = data_x.T
        phi = 2*jnp.pi*phi
        psi = 2*jnp.pi*psi
        # Get result
        result = (
            self.data_k1*jnp.cos(phi - self.data_mu)
            + self.data_k2*jnp.cos(psi - self.data_nu)
            - self.data_k3*jnp.cos(phi - self.data_mu - psi + self.data_nu)
        )
        # Func return
        return result


    def prob(self, data_x):
        """
        Get probability distribution
        """
        # Func return - examine log_prob(input), the input may be inverted
        return jnp.exp(self.log_prob(data_x))


# Dist - GW Metric Density


class TemplateDensity:
    """
    GW template density class for hp based results
    """
    def __init__(self, param_ripple):
        self.data_tc, self.data_phic = param_ripple


    def log_prob(self, data_x):
        '''
        Get the log template density

        Take in 2d param
        Feed 2d param to external func for some density result (dtype: float ish)
        The float is the return of log_prob()
        '''
        # Local assignment, 2-d param
        # data_x.shape (n, 2)
        data_mc, data_eta = data_x.T
        data_mc = data_mc + 1.0 # 1.0 - 2.0
        data_eta = data_eta * 0.24 + 0.01 # 0.05 - 0.25
        # Param build with shape (n, 4)
        param = gw_fim.fim_param_stack(data_mc, data_eta)
        # Get results with shape (n, )
        result = gw_fim.map_density_plus(param)
        # Func return
        return result


    def prob(self, data_x):
        """
        Get probability distribution
        """
        # Func return - examine log_prob(input), the input may be inverted
        return jnp.exp(self.log_prob(data_x))
