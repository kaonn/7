from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from numpy import zeros, array, concatenate, hstack, load, array_equal
import numpy as np
import matplotlib.pyplot as plt
from palettable.cubehelix import Cubehelix

def assemble_input(ctxs, ctxsn, goals, goalsn, start, end):
    m, n, o1, p1 = ctxs.shape
    _, p1, p2 = ctxsn.shape
    _, r1, s1 = goals.shape
    _, r2, s2    = goalsn.shape

    X = zeros((end-start, (o1 * p1 + p1*p2) + r1 * s1 + r2*s2))
    for i in range(start,end):
        ci = flatten(ctxs[i])
        ci = flatten((ci, ctxsn[i]))
        X[i-start] = concatenate((ci, concatenate((flatten(goals[i]), flatten(goalsn[i])))))
    return X

X = assemble_input(load("data/oh_train_ctxs.npz"), load("data/oh_train_ctxs_n.npz"), load("data/oh_train_goals.npz"), load("data/oh_train_goals_n.npz"),0,10000)
#X_embedded = TSNE(n_components=2).fit_transform(X)
#print(X_embedded)
y = load("data/oh_train_labels.npz")

X1 = 0
X_embedded = 0

def cluster(X,y):
    svd = TruncatedSVD(n_components=50, n_iter=5, random_state=42)
    global X1
    X1 = svd.fit_transform(X) 
    print('finished svd')
    global X_embedded
    X_embedded = TSNE(n_components=2).fit_transform(X)
    print('finished tsne')
    palette = Cubehelix.make(start=0.3, rotation=-0.5, n=13).mpl_colors
    colors = palette[np.where(y == 1)[1]]

    fig, ax = plt.subplots()
    ax.scatter(X_embedded, y, c=colors, alpha=0.5)

    ax.set_title('tsne clustering')

    ax.grid(True)
    fig.tight_layout()

    plt.show()
    plt.savefig('cluster.png')

print(X.shape,y.shape)
get_rule = KNeighborsClassifier(n_neighbors=y.shape[1])
get_rule.fit(X, y)

Xt = assemble_input(load("data/oh_test_ctxs.npz"), load("data/oh_test_ctxs_n.npz"), load("data/oh_test_goals.npz"), load("data/oh_test_goals_n.npz"),0,2000)
yt = load("data/oh_test_labels.npz")
print(Xt.shape,yt.shape)

def logistic(Xd,yd,Xt,yt):
    clf = LogisticRegressionCV(cv=10, random_state=0, multi_class='multinomial').fit(Xd, yd)
    print(clf.score(Xt,yt))

def knn(Xt,yt):
    s = 0
    for i in range(Xt.shape[0]):
        if array_equal(get_rule.predict([Xt[i]])[0], yt[i]):
            print(s)
            s += 1

    print("Test error on {} samples is {:.0%} after training on {} samples".format(Xt.shape[0], s / Xt.shape[0], X.shape[0]))
