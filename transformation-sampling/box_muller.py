import numpy as np
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt

# generate
N = 10000
uniform_rvs = uniform(loc=0, scale=1).rvs(N)
u1, u2 = uniform_rvs[:N//2], uniform_rvs[N//2:]

# box muller method to generate Normal Random Numbers
x1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
x2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
norm_rvs = np.concatenate((x1, x2))

# compute the pdf of Normal Distribution
norm_rv = norm(loc=0, scale=1)
x = np.linspace(norm_rv.ppf(0.001), norm_rv.ppf(0.999), 10000)
y = norm_rv.pdf(x)

nbins = 50
plt.hist(norm_rvs, bins=nbins, normed=True, ec='black', label='Random Numbers')
plt.plot(x, y, 'r-', lw=2, label='PDF')
plt.legend()
plt.xlim([norm_rv.ppf(0.001), norm_rv.ppf(0.999)])
plt.show()