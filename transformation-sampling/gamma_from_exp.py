import numpy as np
from scipy.stats import uniform, gamma
import matplotlib.pyplot as plt

# parameter values
shape = 5 # a
scale = 3.0

# generate Uniform Random Numbers
np.random.seed()
N = 10000
uniform_rvs = uniform(loc=0.0, scale=1.0).rvs(size=N * shape)

# generate Exponential Random Numbers using the inversion method
expon_rvs_inv = -scale * np.log(1-uniform_rvs).reshape((N, -1))
gamma_rvs = np.sum(expon_rvs_inv, axis=1)

# compute pdf of Exponential distribution
gamma_rv = gamma(a=shape, scale=scale)
x = np.linspace(gamma_rv.ppf(0.01), gamma_rv.ppf(0.99), 1000)
y = gamma_rv.pdf(x)

# plot a histgram
nbins = 50
plt.hist(gamma_rvs, bins=nbins, normed=True, ec='black', label='Random Numbers')
plt.plot(x, y, 'r-', lw=2, label='PDF')
plt.legend()
plt.xlim((gamma_rv.ppf(0.01), gamma_rv.ppf(0.99)))
plt.show()