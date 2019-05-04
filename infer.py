import pymc3 as pm
from numpy import zeros, ones, identity, array, random, around, linalg, absolute

n = 5
m = 2
N = 10

Gd = random.randint(2, size=(N,n))
Cd = random.randint(2, size=(N,n))
P1d = random.randint(2, size=(N,n))

Gt = random.randint(2, size=(N,n))
Ct = random.randint(2, size=(N,n))
P1t = random.randint(2, size=(N,n))

with pm.Model() as model:
    theta = pm.Dirichlet('theta', ones(n))

    Gamma = pm.Bernoulli('goal', p=theta, shape=n, observed=Gd)
    C = pm.Bernoulli('context', p=theta, shape=n, observed=Cd)

    beta = pm.MvNormal('beta', mu=zeros(m), cov=identity(m), shape=m)

    # Results are degree of adjacency, need to normalize (see below)
    P1 = pm.MvNormal('premise 1', mu=beta[0] * Gamma + beta[1] * C, cov=identity(n), shape=n, \
                     observed=P1d)

    # Solve the MAP estimate
    map = pm.find_MAP()
    beta_s = map['beta']
    theta_s = map['theta']

    # Sample from the conditional on P_i, take average loss across one test point, then across entire hold out set
    for i in range(N):
        Pi = random.normal(beta_s[0] * Gt[i] + beta_s[1] * Ct[i])
        print(around(absolute(Pi) / linalg.norm(Pi)))
        print(Pi)
        print(P1t[i])
