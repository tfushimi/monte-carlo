import numpy as np
from scipy.stats import geom, nbinom
import matplotlib.pyplot as plt

# the number of samples
N = 10000
upper = 30

# degree of freedom
n = 10
p = 0.6

# define random variables
geom_rv = geom(p=p) # this is First Success distribution p(k) = (1-p)^(k-1)p
nbinom_rv = nbinom(n=n, p=p)

# generate random numbers
geom_rvs = geom_rv.rvs(size=n*N).reshape((N, -1)) - 1 # convert Fs(p) into Ge(p) where p(k) = (1-p)^kp
nbinom_rvs = nbinom_rv.rvs(size=N)
nbinom_rvs_from_geom = geom_rvs.sum(axis=1)

# plot histgram
nbins = np.arange(-0.5, upper, 1.0)
weights = np.ones(N) / N
x = np.arange(upper)
plt.hist(nbinom_rvs_from_geom, nbins, weights=weights)
plt.plot(x, nbinom_rv.pmf(x), 'ro-', lw=1)
plt.xlim((0, upper))
plt.show()