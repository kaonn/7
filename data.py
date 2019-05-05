from numpy import zeros, zeros_like, concatenate, ndarray, load, array_equal

# Data from file
ctxs_train = load("data/training_ctxs.npz")
ctxsn_train = load("data/training_ctxs_n.npz")
goals_train = load("data/training_goals.npz")
goalsn_train = load("data/training_goals_n.npz")
premise_goals_train = load("data/training_premise_goals.npz")
premise_goalsn_train = load("data/training_premise_goals_n.npz")

ctxs_test = load("data/test_ctxs.npz")
ctxsn_test = load("data/test_ctxs_n.npz")
goals_test = load("data/test_goals.npz")
goalsn_test = load("data/test_goals_n.npz")
premise_goals_test = load("data/test_premise_goals.npz")
premise_goalsn_test = load("data/test_premise_goals_n.npz")

def cinput(ctxs, ctxsn, goals, goalsn):
    m, n, o = ctxs.shape
    _, p    = ctxsn.shape

    X = zeros((m, 2 * n * o + 2 * p))
    for i in range(m):
        X[i] = concatenate((ndarray.flatten(ctxs[i]), ctxsn[i], \
                            ndarray.flatten(goals[i]), goalsn[i]))
    return X

# Data for classifier
X = cinput(ctxs_train, ctxsn_train, goals_train, goalsn_train)
y = load("data/training_labels.npz")

Xt = cinput(ctxs_test, ctxsn_test, goals_test, goalsn_test)
yt = load("data/test_labels.npz")

def ginput(data, datan):
    m, n, o = data.shape
    _, p    = datan.shape
    X = zeros((m, n * o + p))
    for i in range(m):
        X[i] = concatenate((ndarray.flatten(data[i]), datan[i]))
    return X

def gpremise(goals, goalsn, p):
    m, _, o1, p1 = goals.shape
    _, _, o2, p2 = goalsn.shape
    X = zeros((m, o1 * p1 + o2 * p2))
    for i in range(m):
        X[i] = concatenate((ndarray.flatten(goals[i][p]), ndarray.flatten(goalsn[i][p])))
    return X

# Data for premise generator
Gd = ginput(ctxs_train, ctxsn_train)
Cd = ginput(goals_train, goalsn_train)
Pd = gpremise(premise_goals_train, premise_goalsn_train, 0)

Gt = ginput(ctxs_test, ctxsn_test)
Ct = ginput(goals_test, goalsn_test)
Pt = gpremise(premise_goals_test, premise_goalsn_test, 0)

def data_for_rule(r, G, C, P, y):
    Gn = zeros_like(P)
    Cn = zeros_like(P)
    Pn = zeros_like(P)
    j = 0
    for i in range(Gn.shape[0]):
        if array_equal(y[i], r):
            Gn[j][:G[i].shape[1]] = G[i]
            Cn[j][:C[i].shape[1]] = C[i]
            Pn[j]  = P[i]
            j += 1
    return Gn[:j+1], Cn[:j+1], Pn[:j+1]
