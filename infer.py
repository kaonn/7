import pymc3 as pm
from numpy import zeros, ones, identity, multiply, inf, any, exp
from numpy.linalg import inv
from data import Gd, Cd, Pd, y, Gt, Ct, Pt, yt, data_for_rule
from scipy.stats import truncnorm
import theano.tensor as tt

m = 2
BOUND = 0

def UnitMvNormal(*args, **kwargs):
    v = pm.MvNormal(*args, **kwargs)
    global BOUND
    pm.Potential("bound {}".format(BOUND), tt.switch(multiply.reduce(multiply(0 <= v, v <= 1)), 0, -inf))
    BOUND += 1
    return v

def UMvNormal_logp(x, mean, C):
    if any(x < 0) or any(x > 1):
        return -inf
    else:
        return -0.5*(x-mean).dot(inv(C)).dot(x-mean)

def make_model(Gd, Cd, *P):
    n, = Gd[0].shape
    with pm.Model() as model:
        theta1 = pm.Beta('theta1', alpha=zeros(n), beta=ones(n), shape=n)

        theta2 = pm.Beta('theta2', alpha=zeros(n), beta=ones(n), shape=n)

        Gamma = pm.Bernoulli('goal', p=theta1, shape=n, observed=Gd)

        C = pm.Bernoulli('context', p=theta2, shape=n, observed=Cd)

        # Results are degree of adjacency
        for i, Pi in enumerate(P):
            beta_i = UnitMvNormal('beta {}'.format(i), mu=zeros(m), cov=identity(m), shape=m)

            UnitMvNormal('premise {}'.format(i), mu=beta_i[0] * Gamma + beta_i[1] * C, \
                        cov=identity(n), shape=n, observed=Pi)

        return model

RULE = zeros((13,))
RULE[1] = 1
Gd, Cd, Pd = data_for_rule(RULE, Gd, Cd, Pd, y)
Gt, Ct, Pt = data_for_rule(RULE, Gt, Ct, Pt, yt)

M = 10
N = 10

Gd = Gd[:M,:N]
Cd = Cd[:M,:N]
Pd = Pd[:M,:N]
Gt = Gt[:M,:N]
Ct = Ct[:M,:N]
Pt = Pt[:M,:N]

with make_model(Gd, Cd, Pd):
    # Solve the MAP estimate
    params = pm.find_MAP()
    beta = params['beta 0']

    # Sample from the conditional on P_i, take average loss across one test point, then across entire hold out set
    # TODO sample from this distribution and visualize it in TSNE versus an actual data point
    avg = 0
    for i in range(M):
        n, = Pt[i].shape
        avg += exp(UMvNormal_logp(Pt[i], beta[0] * Gt[i] + beta[1] * Ct[i], identity(n)))

    print("Avg likelihood: {}".format(avg / M))
