from __future__ import division
import numpy as np
import numpy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob

cropped = False
cropped = True

def loadData(files):
    fnames = glob.glob(files)
    if cropped:
        X = np.array([plt.imread(fname) for fname in fnames])
        sh = X[0].shape[0:2]
        return np.array([x[:, :, 0].flatten() for x in X]), sh
    else:
        X = np.array([plt.imread(fname) for fname in fnames])
        sh = X[0].shape[0:2]
        return np.array([x.flatten() for x in X]), sh

files = 'yalefaces_cropBackground/subject*' if cropped else 'yalefaces/subject*'

X, orig_shape = loadData(files)
n, d = X.shape
print 'num files:', n
print 'dim files:', d

mu = np.mean(X, 0)
M = np.outer(np.ones(n), mu)
Xc = X - M
# print 'mean of Xc:', np.mean(Xc, 0)
U, S, V = la.svd(Xc, full_matrices=False)

p = 150
p = 60
# p = 10
Vp = V[0:p, :]
print 'X.shape:', X.shape
print 'Vp.shape:', Vp.shape
Z = X.dot(Vp.T)
Xt = M + Z.dot(Vp)


nrows = 2
ncols = 6
for i in range(nrows):
    for j in range(ncols):
        k = i * ncols + j
        if k >= nrows * ncols - 1:
            break
        if k >= p:
            break
        plt.subplot(nrows, ncols, k)
        plt.imshow(Vp[k].reshape(orig_shape), cmap=cm.Greys_r)
        plt.axis('off')

for i in range(n):
    x, xt = X[i, :], Xt[i, :]

    plt.subplot(nrows, ncols, nrows * ncols - 1)
    plt.imshow(x.reshape(orig_shape), cmap=cm.Greys_r)
    plt.axis('off')
    plt.subplot(nrows, ncols, nrows * ncols)
    plt.imshow(xt.reshape(orig_shape), cmap=cm.Greys_r)
    plt.axis('off')
    plt.waitforbuttonpress()
