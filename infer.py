import pymc3 as pm
from numpy import zeros, ones, identity, multiply, inf
from scipy.stats import multivariate_normal as mvn
from data import Gd, Cd, Pd, y, Gt, Ct, Pt, yt, data_for_rule
import theano.tensor as tt

m = 2
BOUND = 0

def DUMvNormal(*args, **kwargs):
    v = pm.MvNormal(*args, **kwargs)
    global BOUND
    pm.Potential("bound {}".format(BOUND), tt.switch(multiply.reduce(\
        multiply(0 <= v, v <= 1)), 0, -inf))
    BOUND += 1
    return v

def make_model(Gd, Cd, *P):
    n, = Gd[0].shape
    with pm.Model() as model:
        theta1 = pm.Beta('theta1', alpha=zeros(n), beta=ones(n), shape=n)

        theta2 = pm.Beta('theta2', alpha=zeros(n), beta=ones(n), shape=n)

        Gamma = pm.Bernoulli('goal', p=theta1, shape=n, observed=Gd)

        C = pm.Bernoulli('context', p=theta2, shape=n, observed=Cd)

        # Results are degree of adjacency
        for i, Pi in enumerate(P):
            beta_i = DUMvNormal('beta {}'.format(i), mu=zeros(m), cov=identity(m), shape=m)

            DUMvNormal('premise {}'.format(i), mu=beta_i[0] * Gamma + beta_i[1] * C, \
                        cov=identity(n), shape=n, observed=Pi)

        return model

RULE = zeros((13,))
RULE[1] = 1
Gd, Cd, Pd = data_for_rule(RULE, Gd, Cd, Pd, y)
Gt, Ct, Pt = data_for_rule(RULE, Gt, Ct, Pt, yt)
M1 = min(Gd.shape[0], 100)
N1 = min(Gd.shape[1], 100)
M2 = min(Gt.shape[0], 100)
N2 = min(Gt.shape[1], 100)

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

    # Compute average likelihood
    # TODO sample from this distribution and visualize it in TSNE versus an actual data point
    avg = 0
    n, = Pt[0].shape
    I = identity(n)
    mus = beta[0] * Gt + beta[1] * Ct
    for i in range(M2):
        avg += mvn.pdf(Pt[i], mus[i], I)

    print("Avg likelihood: {}".format(avg / M2))
