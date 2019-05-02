from sklearn.neighbors import KNeighborsClassifier
from numpy import zeros, concatenate, hstack, load, array_equal

def assemble_input(goals, goalsn):
    n, m, o = goals.shape
    _, q, r = goalsn.shape
    X = zeros((n, m * o + q * r))
    for i in range(n):
        X[i] = concatenate(hstack((goals[i], goalsn[i])))
    return X

#ctxs = load("data/training_ctxs.npz")
#ctxsn = load("data/training_ctxs_n.npz")
X = assemble_input(load("data/training_goals.npz"), load("data/training_goals_n.npz"))
y = load("data/training_labels.npz")

get_rule = KNeighborsClassifier(n_neighbors=y.shape[1])
get_rule.fit(X, y)

Xt = assemble_input(load("data/test_goals.npz"), load("data/test_goals_n.npz"))
yt = load("data/test_labels.npz")

s = 0
for i in range(Xt.shape[0]):
    yti = get_rule.predict([Xt[i]])[0]
    if array_equal(yti, yt[i]):
        s += 1

print("Test error on {} samples is {:.0%} after training on {} samples".format(Xt.shape[0], s / Xt.shape[0], X.shape[0]))
