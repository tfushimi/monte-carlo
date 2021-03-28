import numpy as np
from scipy.stats import uniform, expon
import matplotlib.pyplot as plt

# generate Uniform Random Numbers
np.random.seed()
N = 10000
uniform_rvs = uniform(loc=0.0, scale=1.0).rvs(size=N)

# generate Exponential Random Numbers using the inversion method
scale = 1.0
expon_rvs_inv = -scale * np.log(1-uniform_rvs)

# compute pdf of Exponential distribution
expon_rv = expon(scale=scale)
x = np.linspace(expon_rv.ppf(0.01), expon_rv.ppf(0.99), 1000)
y = expon_rv.pdf(x)

# plot a histgram
nbins = 50
plt.hist(expon_rvs_inv, bins=nbins, normed=True, ec='black', label='Random Numbers')
plt.plot(x, y, 'r-', lw=2, label='PDF')
plt.legend()
plt.xlim((expon_rv.ppf(0.01), expon_rv.ppf(0.99)))
plt.show()