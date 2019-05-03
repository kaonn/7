from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from numpy import zeros, array, concatenate, hstack, load, array_equal

def assemble_input(ctxs, ctxsn, goals, goalsn):
    m, n, o1, p1 = ctxs.shape
    _, _, o2, p2 = ctxsn.shape
    _, r1, s1    = goals.shape
    _, r2, s2    = goalsn.shape

    X = zeros((m, n * (o1 * p1 + o2 * p2) + r1 * s1 + r2 * s2))
    for i in range(m):
        ci = concatenate(hstack((ctxs[i][0], ctxsn[i][0])))
        for j in range(1, n):
            ci = concatenate((ci, concatenate(hstack((ctxs[i][j], ctxsn[i][j])))))
        X[i] = concatenate((ci, concatenate(hstack((goals[i], goalsn[i])))))
    return X

X = assemble_input(load("data/training_ctxs.npz"), load("data/training_ctxs_n.npz"), load("data/training_goals.npz"), load("data/training_goals_n.npz"))
#X_embedded = TSNE(n_components=2).fit_transform(X)
#print(X_embedded)
y = load("data/training_labels.npz")

get_rule = KNeighborsClassifier(n_neighbors=y.shape[1])
get_rule.fit(X, y)

Xt = assemble_input(load("data/test_ctxs.npz"), load("data/test_ctxs_n.npz"), load("data/test_goals.npz"), load("data/test_goals_n.npz"))
yt = load("data/test_labels.npz")

s = 0
for i in range(Xt.shape[0]):
    if array_equal(get_rule.predict([Xt[i]])[0], yt[i]):
        s += 1

print("Test error on {} samples is {:.0%} after training on {} samples".format(Xt.shape[0], s / Xt.shape[0], X.shape[0]))
