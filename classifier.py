from sklearn.neighbors import KNeighborsClassifier
from numpy import array_equal
from data import X, y, Xt, yt

get_rule = KNeighborsClassifier(n_neighbors=5)
get_rule.fit(X, y)

s = 0
for i in range(Xt.shape[0]):
    if array_equal(get_rule.predict([Xt[i]])[0], yt[i]):
        s += 1

print("Test error on {} samples is {:.0%} after training on {} samples".format(Xt.shape[0], s / Xt.shape[0], X.shape[0]))
