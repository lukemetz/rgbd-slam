import numpy as np

A = np.matrix([
    [1., 0., 0., 2.],
    [0., 1., 0., 1.],
    [0., 0., 1., -2.]])

print A

p = np.matrix([
    [1., 1., 3., 1.],
    [3., 4., 2., 1.],
    [6., 6., 8., 1.],
    [1., 2., 3., 1.]]).transpose()
print p
print "pre"

print "mult"

target = A*p
target[2, 2] += .1
target[2, 3] += .1
print target

print np.round(target * np.linalg.inv(p), 10)
