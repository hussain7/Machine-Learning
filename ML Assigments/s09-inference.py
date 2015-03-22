from __future__ import division
from pdb import set_trace as keyboard
import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt
import datetime

### ============================================================================
# Exercise 2
#

def sample_x(n):
  return r.normal(0, 1, (n,));

def sample_y_x(xpos):
  return r.binomial(1, .9 if xpos else .1)

def sample_xy(n):
  X = sample_x(n)
  Y = np.array([ sample_y_x(x > 0) for x in X ])
  return (X, Y)

def sample_reject_3(n):
  X = np.array([])
  Y = np.array([])
  while n > 0:
    x, y = sample_xy(n)
    X = np.concatenate((X, x))
    Y = np.concatenate((Y, y))
    n -= sum(y)
  return (X, Y)

def ex2():
  # Hist options
  n_y1_samples = 1000000
  histopt = { 'bins':250,
              'normed':1,
              'histtype':'stepfilled'
            }

  X, Y = sample_reject_3(n_y1_samples)
  X_Y0 = X[Y==0]
  X_Y1 = X[Y==1]

  print "Tot Y = 1 samples:", X_Y1.size
  print "Tot Y = 0 samples:", X_Y0.size
  print "Tot any Y samples:", X.size

  E_x_y = X_Y1.mean()
  print "E[x / y = 1] =", E_x_y

  plt.clf()

  plt.subplot(3, 1, 1)
  plt.xlim(-3.0, 3.0)
  plt.hist(X,
            color = 'g',
            **histopt)

  plt.subplot(3, 1, 2)
  plt.xlim(-3.0, 3.0)
  plt.hist(X_Y0,
            color = 'b',
            alpha = 1,
            **histopt)
  plt.hist(X_Y1,
            color = 'r',
            alpha = .6,
            **histopt)

  plt.subplot(3, 1, 3)
  plt.xlim(-3.0, 3.0)
  n, bins, _ = plt.hist(X_Y1,
                      color = 'r',
                      **histopt)
  plt.plot([E_x_y, E_x_y], [0, n[sum(bins < E_x_y)-1]], 'k')
  plt.show()

### ============================================================================
# Exercise 3
#

# Conditional Probability Tables
cp_b = .98 # Not so conditional here =)
cp_f = .95
cp_g = np.array([[.01, .9], [.03, .96]])
cp_t = np.array([.02, .97])
cp_s = np.array([[0, 0], [.08, .99]])

def sample_b(n):
  return r.binomial(1, cp_b, (n,))

def sample_f(n):
  return r.binomial(1, cp_f, (n,))

def sample_g_bf(b, f):
  return np.array([ r.binomial(1, cp_g[bb][ff]) for bb, ff in zip(b, f) ])

def sample_t_b(b):
  return np.array([ r.binomial(1, cp_t[bb]) for bb in b ])

def sample_s_tf(t, f):
  return np.array([ r.binomial(1, cp_s[tt][ff]) for tt, ff in zip(t, f) ])

def likelihood_s0_tf(t, f):
  return np.array([ 1-cp_s[tt][ff] for tt, ff in zip(t, f) ])

def sample_joint(n):
  b = sample_b(n)
  f = sample_f(n)
  g = sample_g_bf(b, f)
  t = sample_t_b(b)
  s = sample_s_tf(t, f)
  return b, f, g, t, s

def sample_rejection(n):
  # Rejection Sampling
  F = np.array([])
  while n > 0:
    b, f, g, t, s = sample_joint(n)
    F = np.concatenate((F, f[s==0]))
    n -= sum(s==0)
  return F

def sample_importance(n):
  # Importance Sampling
  B = sample_b(n)
  F = sample_f(n)
  T = sample_t_b(B)
  W = likelihood_s0_tf(T, F)
  return F, W

def ex3():
  # Hist options
  histopt = { 'bins':[-.5, .5, 1.5],
              'normed':1,
              'histtype':'stepfilled'
            }

  plt.clf()

  Kv = np.array([ 500, 10000, 50000, 100000, 500000 ])
  numK = Kv.size
  for i, n_s0_samples in enumerate(Kv):
    print "K =", n_s0_samples

# Rejection Samplings
    t0 = datetime.datetime.now()
    F = sample_rejection(n_s0_samples)
    t1 = datetime.datetime.now()
    print "Rejection Sampling:  E[F / S = 0] =", sum(F)/n_s0_samples, " (nsecs:", (t1-t0).total_seconds(), ")"

# Importance Sampling
    t0 = datetime.datetime.now()
    F, W = sample_importance(n_s0_samples)
    t1 = datetime.datetime.now()
    print "Importance Sampling: E[F / S = 0] =", sum(W[F==1])/sum(W), " (nsecs:", (t1-t0).total_seconds(), ")"

### ============================================================================
# Main
#

if __name__ == '__main__':
  ex2()
  # ex3()
