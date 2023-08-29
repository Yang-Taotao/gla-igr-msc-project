"""
VI config file.
"""
# Library import
import haiku as hk
from data import vi_cls

# Target distribution. Bivariate von Mises distribution on a 2-Torus.
LOC = [0.0, 0.0]
CONCENTRATION = [4.0, 4.0]
CORRELATION = 0.0
# Target density params tc, phic
PARAM_RIPPLE = [0.0, 0.0]

# Flow parameters
NUM_PARAMS = 2
NUM_FLOW_LAYERS = 2
HIDDEN_SIZE = 8
NUM_MLP_LAYERS = 2
NUM_BINS = 4

# Perform variational inference
TOTAL_EPOCHS = 3000 #reduce this for testing purpose, original val = 10000
NUM_SAMPLES = 1000 #1000
LEARNING_RATE = 0.001 #0.001

# Other cfg
PRNG_SEQ = hk.PRNGSequence(42)

# Target distribution selector
DIST_BVM = vi_cls.BivariateVonMises(LOC, CONCENTRATION, CORRELATION)
DIST_GW = vi_cls.TemplateDensity(PARAM_RIPPLE)
