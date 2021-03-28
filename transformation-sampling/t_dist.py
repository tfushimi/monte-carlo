import numpy as np
from scipy.stats import norm, t, chi2
import matplotlib.pyplot as plt

# the number of samples
N = 10000

# degree of freedom
df = 5

# define random variables
norm_rv = norm(loc=0, scale=1)
chi2_rv = chi2(df=df)
t_rv = t(df=5)

# generate random numbers
norm_rvs = norm_rv.rvs(size=N)
chi2_rvs = chi2_rv.rvs(size=N)
t_rvs = norm_rvs / np.sqrt(chi2_rvs / df)

# plot histgram
nbins = 50
plt.hist(t_rvs, normed=True, bins=nbins, ec='black')
x = np.linspace(t_rv.ppf(0.01), t_rv.ppf(0.99), 1000)
y = t_rv.pdf(x)
plt.plot(x, y, 'r-', lw=2)
plt.xlim((t_rv.ppf(0.01), t_rv.ppf(0.99)))
plt.show()