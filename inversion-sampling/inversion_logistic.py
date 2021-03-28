import numpy as np
from scipy.stats import uniform, logistic
import matplotlib.pyplot as plt

# generate Uniform Random Numbers
np.random.seed()
N = 10000
uniform_rvs = uniform(loc=0.0, scale=1.0).rvs(size=N)

# generate Logistic Random Numbers using the inversion method
mu = 0
s = 1
logistic_rvs_inv = mu + s * np.log(uniform_rvs / (1-uniform_rvs))

# compute the pdf of Logistic distribution
logistic_rv = logistic(loc=mu, scale=s)
x = np.linspace(logistic_rv.ppf(0.01), logistic_rv.ppf(0.99), 10000)
y = logistic_rv.pdf(x)

# plot a histgram
nbins = 50
plt.hist(logistic_rvs_inv, normed=True, bins=nbins, ec='black', label='Random Numbers')
plt.plot(x, y, 'r-', lw=2, label='PDF')
plt.legend()
plt.xlim([logistic_rv.ppf(0.01), logistic_rv.ppf(0.99)])
plt.show()