import numpy as np
from scipy.stats import uniform, beta
from scipy.special import betaincinv
import matplotlib.pyplot as plt

# generate Uniform Random Numbers
np.random.seed()
N = 10000
uniform_rvs = uniform(loc=0.0, scale=1.0).rvs(size=N)

# generate Gamma Random Numbers using the inversion method
a = 0.7
b = 0.7
beta_rvs_inv = betaincinv(a, b, uniform_rvs)

# compute the pdf of Gamma distribution
beta_rv = beta(a=a, b=b)
x = np.linspace(beta_rv.ppf(0.01), beta_rv.ppf(0.99), 10000)
y = beta_rv.pdf(x)

# plot a histgram
nbins = 50
plt.hist(beta_rvs_inv, normed=True, bins=nbins, ec='black', label='Random Numbers')
plt.plot(x, y, 'r-', lw=2, label='PDF')
plt.legend()
plt.xlim([beta_rv.ppf(0.01), beta_rv.ppf(0.99)])
plt.show()