import pymc3 as pm
from numpy import zeros, ones, zeros_like, ones_like, identity, multiply, log
from numpy.linalg import inv
from data import Gd, Cd, Pd, y, Gt, Ct, Pt, yt, data_for_rule

import theano.tensor as tt
from theano.tensor import _shared as te
from theano.compile.ops import as_op

#from scipy.stats import multivariate_normal as mvn
from scipy.integrate import nquad

m = 2

#def tzip(a, b):
#    return list((a[i], b[i]) for i in range(a.shape[0]))

# Truncated multivariate normal distribution (TMvN)
@as_op(itypes=[tt.dvector, tt.dvector, tt.dmatrix, tt.dvector, tt.dvector], otypes=[tt.dscalar])
def TMvNormal_pdf(x, mu, cov, a, b):
    #if any(x < a - eps) or any(x > b + eps): return 0
    # Unnormalized MvNormal pdf
    def updf(x):
        dx = x - mu
        return exp((-1 / 2) * dx.T.dot(inv(cov).dot(dx)))
    #pdf = lambda x: mvn.pdf(x, mu=mu, cov=cov)
    return updf(x) / nquad(updf, zip(a, b))

@as_op(itypes=[tt.dvector, tt.dvector, tt.dmatrix, tt.dvector, tt.dvector], otypes=[tt.dscalar])
def TMvNormal_cdf(X, mu, cov, a, b):
    return nquad(lambda x: TMvNormal_pdf(x, mu, cov, a, b), zip(zeros_like(a), X))

# Discretization trick
def DTMvNormal_pdf(x, mu, cov, a, b):
    cdf = lambda x: TMvNormal_cdf(x, mu, cov, a, b)
    return cdf(x + ones_like(x)) - cdf(x)

def DTMvNormal(name, mu, cov, a, b, **kwargs):
    return pm.DensityDist(name, logp=lambda x: log(DTMvNormal_pdf((x), (mu), (cov), (a), (b))), **kwargs)

# Unit DTMvN
def UDTMvNormal(name, mu, cov, **kwargs):
    n, = mu.shape
    return DTMvNormal(name, mu, cov, zeros(n), ones(n), **kwargs)

def UDTMvNormal_pdf(x, mu, cov):
    return DTMvNormal_pdf(x, mu, cov, zeros_like(x), ones_like(x))

def make_model(Gd, Cd, *P):
    n, = Gd[0].shape
    with pm.Model() as model:
        theta1 = pm.Beta('theta1', alpha=zeros(n), beta=ones(n), shape=n)

        theta2 = pm.Beta('theta2', alpha=zeros(n), beta=ones(n), shape=n)

        Gamma = pm.Bernoulli('goal', p=theta1, shape=n, observed=Gd)

        C = pm.Bernoulli('context', p=theta2, shape=n, observed=Cd)

        # Results are degree of adjacency
        for i, Pi in enumerate(P):
            beta_i = UDTMvNormal('beta {}'.format(i), mu=zeros(m), cov=identity(m), shape=m)

            UDTMvNormal('premise {}'.format(i), mu=beta_i[0] * Gamma + beta_i[1] * C, \
                        cov=identity(n), shape=n, observed=Pi)

        return model

RULE = zeros((13,))
RULE[1] = 1
Gd, Cd, Pd = data_for_rule(RULE, Gd, Cd, Pd, y)
Gt, Ct, Pt = data_for_rule(RULE, Gt, Ct, Pt, yt)
M1 = min(Gd.shape[0], 10)
N1 = min(Gd.shape[1], 10)
M2 = min(Gt.shape[0], 10)
N2 = min(Gt.shape[1], 10)

Gd = Gd[:M1,:N1]
Cd = Cd[:M1,:N1]
Pd = Pd[:M1,:N1]
Gt = Gt[:M2,:N2]
Ct = Ct[:M2,:N2]
Pt = Pt[:M2,:N2]

with make_model(Gd, Cd, Pd):
    # Solve the MAP estimate
    params = pm.sample()
    beta = params['beta 0']

    # Sample from the conditional on P_i, take average loss across one test point, then across entire hold out set
    # TODO sample from this distribution and visualize it in TSNE versus an actual data point
    avg = 0
    for i in range(M2):
        n, = Pt[i].shape
        avg += UDTMvNormal_pdf(Pt[i], beta[0] * Gt[i] + beta[1] * Ct[i], identity(n))

    print("Avg likelihood: {}".format(avg / M2))
