import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import uniform, beta

np.random.seed()

N = 10000
beta_rv = beta(a=2.7, b=6.3)

M = minimize_scalar(fun=lambda x : -beta_rv.pdf(x)).x

# sample from the proposal dist
proposal_dist = uniform(loc=0, scale=1)
# U ~ Uniform(0, 1) for x axis
u_x = proposal_dist.rvs(size=N)
# U ~ Uniform(0, M) for y axis
u_y = uniform(loc=0, scale=M).rvs(size=N)

# accept U if U <= f(x)
beta_rvs = u_x[u_y <= beta_rv.pdf(u_x)]

# compute acceptance ratio
print('M is {0:.3f}, Acceptance ratio is {1:.3f}, 1/M is {2:.3f}'.format(M, len(beta_rvs)/N, 1/M))

# plot histogram
nbins = 50
lower, upper = beta_rv.ppf(0.01), beta_rv.ppf(0.99)
plt.hist(beta_rvs, bins=nbins, normed=True, ec='black')
x = np.linspace(lower, upper, 1000)
plt.plot(x, beta_rv.pdf(x), 'r-', lw=2)
plt.xlim(lower, upper)
plt.show()
