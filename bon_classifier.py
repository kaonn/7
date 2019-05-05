from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from numpy import zeros, array, concatenate, hstack, load, array_equal
import numpy as np
import matplotlib.pyplot as plt
from palettable.cubehelix import Cubehelix
import pickle

def assemble_input(ctxs, ctxsn, goals, goalsn,start,end):
    m, n, o1, p1 = ctxs.shape
    _, n1, p1, p2 = ctxsn.shape
    _, r1, s1 = goals.shape
    _, r2, s2    = goalsn.shape

    X = zeros((end-start, (n*o1 * p1 + n1*p1*p2) + r1 * s1 + r2*s2))
    for i in range(start,end):
        ci = ctxs[i].flatten()
        ci = concatenate((ci, ctxsn[i].flatten()))
        X[i-start] = concatenate([ci, goals[i].flatten(), goalsn[i].flatten()])
    return X
X = assemble_input(load("oh_train_ctxs.npz"), load("oh_train_ctxs_n.npz"), load("oh_train_goals.npz"), load("oh_train_goals_n.npz"),0,5000)
#X_embedded = TSNE(n_components=2).fit_transform(X)
#print(X_embedded)
y = load("oh_train_labels.npz")
colors = [np.nonzero(y[i])[0][0] for i in range(y.shape[0])]
unique, counts = np.unique(colors, return_counts=True)
d = dict(zip(unique, counts))
print(d)

'''
X1 = 0
X_embedded = 0
'''
def cluster(X,y,dim):
    print('clustering')
    svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
    X1 = svd.fit_transform(X) 
    print('finished svd')
    if dim == 2:
        X_embedded = TSNE(n_components=2, perlexity=50).fit_transform(X1)
        print('finished tsne')
        palette = Cubehelix.make(start_hue=-300., end_hue=240.,
                                           min_sat=1., max_sat=2.5,
                                           min_light=0.3, max_light=0.8, gamma=.9,
                                           n=10).mpl_colors
        colors = [palette[np.nonzero(y[i])[0][0]] for i in range(y.shape[0])]

        fig, ax = plt.subplots()
        #ax.scatter(X1[:,0], X1[:,1], c=colors, alpha=0.5)
        ax.scatter(X_embedded[:,0], X_embedded[:,1], c=colors, alpha=0.5)

        ax.set_title('tsne clustering')

        ax.grid(True)
        fig.tight_layout()

        plt.show()
        plt.savefig('tsne_cluster1.png')

    if dim == 3:
        X_embedded = TSNE(n_components=3, perlexity=50).fit_transform(X1)
        print('finished tsne')
        palette = Cubehelix.make(start_hue=-300., end_hue=240.,
                                           min_sat=1., max_sat=2.5,
                                           min_light=0.3, max_light=0.8, gamma=.9,
                                           n=10).mpl_colors
        colors = [palette[np.nonzero(y[i])[0][0]] for i in range(y.shape[0])]

        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        #ax.scatter(X1[:,0], X1[:,1], c=colors, alpha=0.5)
        ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2], c=colors, alpha=0.5)

        ax.set_title('tsne clustering')

        ax.grid(True)
        fig.tight_layout()

        plt.show()
        plt.savefig('tsne_cluster1.png')
        pickle.dump(fig, open('FigureObject.fig.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`


print(X.shape,y.shape)

Xt = assemble_input(load("oh_test_ctxs.npz"), load("oh_test_ctxs_n.npz"), load("oh_test_goals.npz"), load("oh_test_goals_n.npz"),0,1000)
yt = load("oh_test_labels.npz")
print(Xt.shape,yt.shape)

def logistic(Xd,yd,Xt,yt):
    clf = LogisticRegressionCV(cv=10, random_state=0, multi_class='multinomial').fit(Xd, yd)
    print(clf.score(Xt,yt))

def knn(Xt,yt):
    get_rule = KNeighborsClassifier(n_neighbors=y.shape[1])
    get_rule.fit(X, y)
    s = 0
    for i in range(Xt.shape[0]):
        if array_equal(get_rule.predict([Xt[i]])[0], yt[i]):
            print(s)
            s += 1

    print("Test error on {} samples is {:.0%} after training on {} samples".format(Xt.shape[0], s / Xt.shape[0], X.shape[0]))
