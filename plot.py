import numpy as np
import matplotlib.pylab as plt

data = np.loadtxt(open("plot.csv","rb"),delimiter=",")
plt.plot(data)
plt.ylim([0, 10])
plt.show()
