from __future__ import division
import itertools
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

# Loads data-files
def loadData(fname):
  D = np.loadtxt(fname)
  X = D[:, 0:-1]
  y = D[:, -1]
  return (X, y)

# Makes various types of feature vectors
# 'lin' -> linear
# 'quad' -> quadratic
# 'poly' -> polynomial up to degree <degrees>
def makeFV(X, feats, degrees = None):
  if feats == 'lin':
    return np.append(np.ones((np.size(X, 0), 1)), X, 1)
  elif feats == 'quad':
    xlin = makeFV(X, 'lin')
    dim = np.size(X, 1)
    ir, ic = np.triu_indices(dim)
    xquad = np.array([ np.outer(x, x)[ir, ic] for x in X ])
    return np.append(xlin, xquad, 1)
  elif feats == 'poly':
    if degrees <= 3:
      xpoly_prev = makeFV(X, 'quad')
    else:
      xpoly_prev = makeFV(X, 'poly', degrees-1)
    xpoly = [[ np.prod(i) for i in itertools.combinations_with_replacement(x, degrees) ] for x in X ]
    return np.append(xpoly_prev, xpoly, 1)

  raise Exception('Feats not implemented yet')

# Solves the ridge-regression problem (computes optimal parameters)
def ridgeRegression(X, y, l = 0):
  n = np.size(X, 1)
  I = np.identity(n)
  I[0, 0] = 0
  XT = X.transpose()
  return la.inv(XT.dot(X)+l*I).dot(XT).dot(y)

# Mean-Squared-Error evaluated on some data, given some parameters beta
def MSE(X, y, B):
  e = X.dot(B) - y
  return e.dot(e) / len(e)

# Plots the data, and the model fitted to the data if <feats> is also given
def plotDataAndModel(X, y, feats = None, degrees = None):
  n, d = np.shape(X)
  if feats != None:
    Xfv = makeFV(X, feats, degrees)
    B = ridgeRegression(Xfv, y)
  if d == 1:
    plt.plot(X, y, 'o')
    plt.show(block = False)
    if feats != None:
      xhat = np.mgrid[X.min():X.max():.1]
      xhat = xhat.reshape((np.size(xhat), 1))
      xhatfv = makeFV(xhat, feats, degrees)
      yhat = xhatfv.dot(B)
      plt.plot(xhat, yhat)
  elif d == 2:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(X[:, 0], X[:, 1], y, marker = 'o')
    if feats != None:
      xmin = X.min(0)
      xmax = X.max(0)

      x0grid, x1grid = np.mgrid[xmin[0]:xmax[0]:.3, xmin[1]:xmax[1]:.3]

      xdim0, xdim1 = np.shape(x0grid)
      xsize = np.size(x0grid)

      x0hat = x0grid.flatten()
      x1hat = x1grid.flatten()
      x0hat = x0hat.reshape((np.size(x0hat), 1))
      x1hat = x1hat.reshape((np.size(x1hat), 1))
      xhat = np.append(x0hat, x1hat, 1)
      xhatfv = makeFV(xhat, feats, degrees)
      yhat = xhatfv.dot(B)
      ygrid = yhat.reshape((xdim0, xdim1))
      ax.plot_wireframe(x0grid, x1grid, ygrid)
      ax.auto_scale_xyz([xmin[0], xmax[0]], [xmin[1], xmax[1]], [y.min(), y.max()])
  else:
    raise Exception("Dimensionality of data not handled.")

# Runs cross-validation with one specific value of lambda
def singleCrossValidation(X, y, feats, degrees, k, l):
  n, d = np.shape(X)
  boundaries = np.linspace(0, n, k+1)
  boundaries = boundaries[1:]
  nind = np.array([ np.sum(i > boundaries) for i in range(n) ])
  bind = np.array([ nind == ki for ki in range(k) ])

  loss = np.empty(k)
  for ki in range(k):
    Xtrain, Xtest = X[~bind[ki]], X[bind[ki]]
    ytrain, ytest = y[~bind[ki]], y[bind[ki]]

    Xtrainfv = makeFV(Xtrain, feats, degrees)
    Xtestfv = makeFV(Xtest, feats, degrees)

    B = ridgeRegression(Xtrainfv, ytrain, l)
    loss[ki] = MSE(Xtestfv, ytest, B)

  return [loss.mean(), loss.var()]

# Runs cross-validation for many values of lambda and plots the errors
def fullCrossValidation(X, y, feats, degrees, k = 10):
  lambdas = np.array([ j*10**i for i in np.arange(-2, 5) for j in [1, 3] ])
  loglambdas = np.log(lambdas)

  Xfv = makeFV(X, feats, degrees)
  betas = np.array([ ridgeRegression(Xfv, y, l) for l in lambdas ])
  loss = np.array([ MSE(Xfv, y, B) for B in betas ])
  cvloss = np.array([ singleCrossValidation(X, y, feats, degrees, k, l) for l in lambdas ])

  plt.figure()
  # Removed variance from plot to avoid fucking the scale up
  #plt.errorbar(loglambdas, cvloss[:, 0], yerr = cvloss[:, 1], fmt = '--')
  plt.errorbar(loglambdas, cvloss[:, 0], fmt = '--')
  plt.plot(loglambdas, loss)

############################
# Beginning of the program #
############################

# Last uncommented line determines data-file to load
dataFname = 'dataLinReg1D.txt'
#dataFname = 'dataLinReg2D.txt'
#dataFname = 'dataQuadReg1D.txt'
dataFname = 'dataQuadReg2D.txt'
dataFname = 'dataQuadReg2D_noisy.txt'

# Last uncommented line determines feature type (<degrees> only for 'poly')
#feats = None
feats = 'lin'
#feats = 'quad'
#feats = 'poly'
degrees = 6

X, y = loadData(dataFname);

# Uncomment to visualize data and fit model
plotDataAndModel(X, y, feats, degrees)

# Uncomment to run cross-validation and find best value of lambda
k = 10
#fullCrossValidation(X, y, feats, degrees, k)

plt.show()
