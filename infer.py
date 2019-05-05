import pymc3 as pm
from numpy import zeros, ones, random, identity
from data import Gd, Cd, Pd, y, data_for_rule

m = 2

def make_model(Gd, Cd, *P):
    n, = Gd[0].shape
    with pm.Model() as model:
        theta = pm.Beta('theta', alpha=zeros(n), beta=ones(n), shape=n)
        Gamma = pm.Categorical('goal', p=theta, shape=n, observed=Gd)
        C = pm.Categorical('context', p=theta, shape=n, observed=Cd)
        beta = pm.MvNormal('beta', mu=zeros(m), cov=identity(m), shape=m)
        
        # Results are degree of adjacency
        for i, Pi in enumerate(P):
            pm.MvNormal('premise {}'.format(i), mu=beta[0] * Gamma + beta[1] * C, cov=identity(n), shape=n, observed=Pi)

        return model

ABS = zeros((13,))
ABS[3] = 1
Gd5, Cd5, Pd5 = data_for_rule(ABS, Gd, Cd, Pd, y)

with make_model(Gd5, Cd5, Pd5):
    # Solve the MAP estimate
    params = pm.find_MAP()
    beta = params['beta']
    theta = params['theta']

    print(beta, theta)
    # Sample from the conditional on P_i, take average loss across one test point, then across entire hold out set
    # TODO sample from this distribution and visualize it in TSNE versus an actual data point
    #avg = 0
    #for i in range(N):
    #    avg += random.normal(x = beta_s[0] * Gt[i] + beta_s[1] * Ct[i])

    #print("Avg likelihood: {}".format(avg / N))
