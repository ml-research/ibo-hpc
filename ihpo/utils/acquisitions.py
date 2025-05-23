import numpy as np
from scipy.stats import norm

def EI(incumbent, mu, stddev):
    
    mu             = mu.reshape(-1,)
    stddev         = stddev.reshape(-1,)

    with np.errstate(divide='warn'):
        imp = mu - incumbent
        Z = imp / stddev
        score = imp * norm.cdf(Z) + stddev * norm.pdf(Z)

    return score