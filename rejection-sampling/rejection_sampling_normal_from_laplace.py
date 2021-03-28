import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.stats import uniform, norm, laplace


np.random.seed()

N = 10000

# define target distribution f(x)
target_rv = norm()

# define proposal distribution g(x)
proposal_rv = laplace()

# find M = sup_x f(x)/g(x)
sol = fmin(lambda x : -target_rv.pdf(x) / proposal_rv.pdf(x), 0.0, disp=False)[0]
M = target_rv.pdf(sol) / proposal_rv.pdf(sol)

# sample from the proposal dist
proposal_x = proposal_rv.rvs(size=N) # proposal_x ~ beta(a, b) for x axis
u = uniform(loc=0, scale=1).rvs(size=N) # u_y ~ Uniform(0, 1) for y axis

# accept u_y if u_y <= f(u_x)
target_rvs = proposal_x[u * M * proposal_rv.pdf(proposal_x) <= target_rv.pdf(proposal_x)]

# compute acceptance ratio
print('M is {0:.3f}, Acceptance ratio is {1:.3f}, 1/M is {2:.3f}'.format(M, len(target_rvs)/N, 1/M))

# plot histogram
lower, upper = target_rv.ppf(0.01), target_rv.ppf(0.99)
plt.hist(target_rvs, bins=50, density=True, ec='black')
x = np.linspace(lower, upper, 1000)
plt.plot(x, target_rv.pdf(x), 'r-', lw=2, label='Target')
plt.plot(x, M*proposal_rv.pdf(x), 'g-', lw=2, label='Proposal')
plt.legend()
plt.xlim(lower, upper)
plt.show()
