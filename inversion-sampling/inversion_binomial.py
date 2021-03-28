import numpy as np
from scipy.stats import uniform, binom
from bisect import bisect
import matplotlib.pyplot as plt

# Binomial parameter
n = 10
p = 0.7

# Upper limit to be computed
upper = 15

# the number of samples
N = 10000

# define a Poisson random variable
rv = binom(n=n, p=p)

# compute cdf
x = np.arange(upper)
prob = rv.cdf(x)

# generate Poisson random numbers from uniform random numbers
binom_rvs = [bisect(prob, u) for u in uniform(loc=0, scale=1).rvs(N)]

# hist()のnormed=Trueはバーの積分が1になる確率密度関数になるため離散分布では使えない
# 離散分布ではバーの高さの合計が1になる確率質量関数にする必要がある
# http://stackoverflow.com/questions/3866520/plotting-histograms-whose-bar-heights-sum-to-1-in-matplotlib
nbins = np.arange(-0.5, upper, 1.0)
weights = np.ones(N) / N
plt.hist(binom_rvs, nbins, weights=weights)
plt.plot(x, rv.pmf(x), 'ro-', lw=1)
plt.xlim((0, upper))
plt.show()