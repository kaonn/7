import pymc3 as pm
from numpy import zeros, ones, random, identity, exp
from data import Gd, Cd, Pd, y, Gt, Ct, Pt, yt, data_for_rule

m = 2

def UnitMvNormal(n):
    return pm.Bound(pm.MvNormal, lower=zeros(n), upper=ones(n))

def make_model(Gd, Cd, *P):
    n, = Gd[0].shape
    with pm.Model() as model:
        theta1 = pm.Beta('theta1', alpha=zeros(n), beta=ones(n), shape=n)

        theta2 = pm.Beta('theta2', alpha=zeros(n), beta=ones(n), shape=n)

        Gamma = pm.Bernoulli('goal', p=theta1, shape=n, observed=Gd)

        C = pm.Bernoulli('context', p=theta2, shape=n, observed=Cd)

        # Results are degree of adjacency
        for i, Pi in enumerate(P):
            beta_i = UnitMvNormal(m)('beta {}'.format(i), mu=zeros(m), cov=identity(m), shape=m)

            pm.MvNormal('premise {}'.format(i), mu=beta_i[0] * Gamma + beta_i[1] * C, \
                        cov=identity(n), shape=n, observed=Pi)
            #pm.Potential("bound", tt.switch((beta_i[0] > 0)*(beta_i[1] > 0), _, -np.inf))

        return model

ABS = zeros((13,))
ABS[3] = 1
Gd, Cd, Pd = data_for_rule(ABS, Gd, Cd, Pd, y)
Gt, Ct, Pt = data_for_rule(ABS, Gt, Ct, Pt, yt)

N = 10

import theano as tt

with make_model(Gd[:N,:N], Cd[:N,:N], Pd[:N,:N]):
    # Solve the MAP estimate
    params = pm.find_MAP()
    beta = params['beta 0']

    # Sample from the conditional on P_i, take average loss across one test point, then across entire hold out set
    # TODO sample from this distribution and visualize it in TSNE versus an actual data point
    avg = 0
    for i in range(min(Pt.shape[0], N)):
        n, = Pt[i,:N].shape
        dist = UnitMvNormal(n).dist(mu=beta[0] * Gt[i,:N] + beta[1] * Ct[i,:N], cov=identity(n), shape=n)
        avg += exp(dist.logp(tt.tensor._shared(Pt[i,:N])).eval())

    print("Avg likelihood: {}".format(avg / N))
