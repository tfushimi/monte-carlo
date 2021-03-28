import numpy as np
from scipy.stats import uniform, poisson
from bisect import bisect
import matplotlib.pyplot as plt

# Poisson parameter
mu = 10

# Upper limit to be computed
upper = 40

# the number of samples
N = 10000

# define a Poisson random variable
rv = poisson(mu=mu)

# compute cdf
x = np.arange(upper)
prob = rv.cdf(x)

# generate Poisson random numbers from uniform random numbers
poisson_rvs = [bisect(prob, u) for u in uniform(loc=0, scale=1).rvs(N)]

# hist()のnormed=Trueはバーの積分が1になる確率密度関数になるため離散分布では使えない
# 離散分布ではバーの高さの合計が1になる確率質量関数にする必要がある
# http://stackoverflow.com/questions/3866520/plotting-histograms-whose-bar-heights-sum-to-1-in-matplotlib
nbins = np.arange(-0.5, upper, 1.0)
weights = np.ones(N) / N
plt.hist(poisson_rvs, nbins, weights=weights)
plt.plot(x, rv.pmf(x), 'ro-', lw=1)
plt.xlim((0, upper))
plt.show()