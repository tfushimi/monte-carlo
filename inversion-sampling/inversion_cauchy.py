import numpy as np
from scipy.stats import uniform, cauchy
import matplotlib.pyplot as plt

# generate Uniform Random Numbers
np.random.seed()
N = 10000
uniform_rvs = uniform(loc=0.0, scale=1.0).rvs(size=N)

# generate Exponential Random Numbers using the inversion method
x0 = 0
gamma = 1
lower, upper = -10, 10
cauchy_rvs_inv = x0 + gamma * np.tan(np.pi*(uniform_rvs-0.5))
cauchy_rvs_inv = cauchy_rvs_inv[(lower <cauchy_rvs_inv) & (cauchy_rvs_inv<upper)]

# compute the pdf of Cauchy distribution
cauchy_rv = cauchy(loc=x0, scale=gamma)
x = np.linspace(lower, upper, 10000)
y = cauchy_rv.pdf(x)

# plot a histgram
nbins = 50
plt.hist(cauchy_rvs_inv, normed=True, bins=nbins, ec='black', label='Random Numbers')
plt.plot(x, y, 'r-', lw=2, label='PDF')
plt.legend()
plt.xlim([lower, upper])
plt.show()