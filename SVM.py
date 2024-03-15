import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
from sklearn.inspection import DecisionBoundaryDisplay


# import some data to play with
## iris = datasets.load_iris()
## # Take the first two features. We could avoid this by using a two-dim dataset
## X = iris.data[:, :2]
## y = iris.target
#dementia_class = {}
#dementia_class["non_demo"] = np.loadtxt("non_v2.txt")
#dementia_class["mod_demo"] = np.loadtxt("moderate_v2.txt")
#
#
#target_balance = {}
#X = np.empty((0))
#Y = np.empty((0))
#for idx, key_s in enumerate(dementia_class):
#    X = np.concatenate((X, dementia_class[key_s]))
#    Y = np.concatenate((Y, [idx] * len(dementia_class[key_s])))
#    target_balance[idx] = key_s

X1 = np.loadtxt('non.txt')[0:300,0:2]
X2 = np.loadtxt('mod.txt')[:,0:2]
X3 = np.loadtxt('mild.txt')[0:200,0:2]
X4 = np.loadtxt('very_mild.txt')[0:200,0:2]
y1 = np.zeros([X1.shape[0]])
y2 = np.ones([X2.shape[0]])
y3 = 2*np.ones([X3.shape[0]])
y4 = 3*np.ones([X4.shape[0]])

X = np.append(X1,X2,axis=0)
X = np.append(X,X3,axis=0)
X = np.append(X,X4,axis=0)
y = np.append(y1,y2)
y = np.append(y,y3)
y = np.append(y,y4)

#print(target_balance)
print(X, y)

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (
    svm.SVC(kernel="linear", C=C),
#    svm.LinearSVC(C=C, max_iter=10000),
#    svm.SVC(kernel="rbf", gamma=0.7, C=C),
 #   svm.SVC(kernel="poly", degree=2, gamma="auto", C=C),
)
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = (
    "SVC with linear kernel",
#    "LinearSVC (linear kernel)",
#   "SVC with RBF kernel",
#    "SVC with polynomial (degree 3) kernel",
)

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots()
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
ax = sub
#for clf, title in zip(models, titles):
#    disp = DecisionBoundaryDisplay.from_estimator(
#        clf,
#        X,
#        response_method="predict",
#        cmap=plt.cm.coolwarm,
#        alpha=0.8,
#        ax=ax,
#        xlabel='n',
#        ylabel='asymmetry',
#    )
#    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
#    ax.set_xticks(())
#    ax.set_yticks(())
#    ax.set_title(title)
#



for clf, title in zip(models, titles):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel='n',
        ylabel='asymmetry',
    )
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
plt.savefig("svm_pca_part3.pdf")
plt.show()
