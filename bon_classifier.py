from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from numpy import zeros, array, concatenate, hstack, load, array_equal
import numpy as np
import matplotlib.pyplot as plt
from palettable.cubehelix import Cubehelix
import pickle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.externals.joblib import parallel_backend

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

def load_input(n,m):
    X = assemble_input(load("oh3_train_ctxs.npz"), load("oh3_train_ctxs_n.npz"), load("oh3_train_goals.npz"), load("oh3_train_goals_n.npz"),0,n)
    y = load("oh3_train_labels.npz")[0:n]
    colors = [np.nonzero(y[i])[0][0] for i in range(y.shape[0])]
    unique, counts = np.unique(colors, return_counts=True)
    d = dict(zip(unique, counts))
    print(d)
    print(X.shape,y.shape)

    Xt = assemble_input(load("oh3_test_ctxs.npz"), load("oh3_test_ctxs_n.npz"), load("oh3_test_goals.npz"), load("oh3_test_goals_n.npz"),0,m)
    yt = load("oh3_test_labels.npz")[0:m]
    print(Xt.shape,yt.shape)
    return X,y,Xt,yt

def cluster(X,y,dim,name):
    print('clustering')
    svd = TruncatedSVD(n_components=50, n_iter=3, random_state=42)
    X1 = svd.fit_transform(X) 
    print('finished svd')
    if dim == 2:
        fig, axes = plt.subplots(3,5)
        for i,perp in enumerate([10,30,50]):
            for j,met in enumerate(['euclidean', 'cityblock', 'hamming', 'jaccard', 'cosine']):

                X_embedded = TSNE(n_components=2, perplexity=perp, metric=met).fit_transform(X1)
                print('finished tsne')
                palette = Cubehelix.make(start_hue=-300., end_hue=240.,
                                                   min_sat=1., max_sat=2.5,
                                                   min_light=0.3, max_light=0.8, gamma=.9,
                                                   n=10).mpl_colors
                colors = [palette[np.nonzero(y[i])[0][0]] for i in range(y.shape[0])]

                #ax.scatter(X1[:,0], X1[:,1], c=colors, alpha=0.5)
                axes[i,j].scatter(X_embedded[:,0], X_embedded[:,1], c=colors, alpha=0.5)

                axes[i,j].set_title(met)

                axes[i,j].grid(True)

        fig.tight_layout()

        plt.show()
        plt.savefig(name+'.png')

    if dim == 3:
        X_embedded = TSNE(n_components=3, perplexity=50).fit_transform(X1)
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
        plt.savefig('tsne_cluster2.png')
        pickle.dump(fig, open('FigureObject.fig.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`


def logistic(Xd,yd,Xt,yt):
    yd1 = [np.nonzero(yd[i])[0][0] for i in range(yd.shape[0])]
    clf = LogisticRegressionCV(cv=10, random_state=0, multi_class='multinomial').fit(Xd, yd1)
    yt1 = [np.nonzero(yt[i])[0][0] for i in range(yt.shape[0])]
    print(clf.score(Xt,yt1))

def knn(X,y,Xt,yt,dist):
    get_rule = KNeighborsClassifier(n_neighbors=y.shape[1], metric=dist)
    get_rule.fit(X, y)
    s = 0
    for i in range(Xt.shape[0]):
        if array_equal(get_rule.predict([Xt[i]])[0], yt[i]):
            print(s)
            s += 1
    print("Test error on {} samples is {:.0%} after training on {} samples".format(Xt.shape[0], s / Xt.shape[0], X.shape[0]))

def bestKNN(X,y,Xt,yt):
    
    clf = GridSearchCV(KNeighborsClassifier(), {'n_neighbors' : [5,8,13], 'metric' : ['euclidean', 'hamming', 'dice', 'jaccard']}, scoring='accuracy', cv=5)
    with parallel_backend('threading',n_jobs=24): 
        clf.fit(X, y)
    results = clf.cv_results_
    print(results)
    
    acc = clf.score(Xt,yt)
    print(acc)
    return results


