import numpy as np
from scipy.stats import uniform, chi2
import matplotlib.pyplot as plt

# parameter values
df = 10
shape = df // 2 # a

# generate Uniform Random Numbers
np.random.seed()
N = 10000
uniform_rvs = uniform(loc=0.0, scale=1.0).rvs(size=N * shape)

# generate Exponential Random Numbers using the inversion method
expon_rvs_inv = - 2 * np.log(1-uniform_rvs).reshape((N, -1))
chi2_rvs = np.sum(expon_rvs_inv, axis=1)

# compute pdf of Exponential distribution
chi2_rv = chi2(df=df)
x = np.linspace(chi2_rv.ppf(0.01), chi2_rv.ppf(0.99), 1000)
y = chi2_rv.pdf(x)

# plot a histgram
nbins = 50
plt.hist(chi2_rvs, bins=nbins, normed=True, ec='black', label='Random Numbers')
plt.plot(x, y, 'r-', lw=2, label='PDF')
plt.legend()
plt.xlim((chi2_rv.ppf(0.01), chi2_rv.ppf(0.99)))
plt.show()