import numpy as np
from scipy.stats import uniform, gamma
from scipy.special import gammainccinv
import matplotlib.pyplot as plt

# generate Uniform Random Numbers
np.random.seed()
N = 10000
uniform_rvs = uniform(loc=0.0, scale=1.0).rvs(size=N)

# generate Gamma Random Numbers using the inversion method
k = 2
theta = 2
gamma_rvs_inv = theta * gammainccinv(k, uniform_rvs)

# compute the pdf of Gamma distribution
gamma_rv = gamma(a=k, scale=theta)
x = np.linspace(gamma_rv.ppf(0.01), gamma_rv.ppf(0.99), 10000)
y = gamma_rv.pdf(x)

# plot a histgram
nbins = 50
plt.hist(gamma_rvs_inv, normed=True, bins=nbins, ec='black', label='Random Numbers')
plt.plot(x, y, 'r-', lw=2, label='PDF')
plt.legend()
plt.xlim([gamma_rv.ppf(0.01), gamma_rv.ppf(0.99)])
plt.show()