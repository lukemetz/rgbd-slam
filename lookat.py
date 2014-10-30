import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "/Users/scope/slam_data/rgbd_dataset_freiburg1_rpy/"
rgb_path = path + "rgb.txt"
depth_path = path + "depth.txt"

rgbs = pd.read_csv(rgb_path, sep=" ", skiprows=[0, 1, 2], header=None, names=["timestamp", "path"])
depths = pd.read_csv(depth_path, sep=" ", skiprows=[0, 1, 2], header=None, names=["timestamp", "path"])

r_times = rgbs["timestamp"].values
d_times = depths["timestamp"].values

lookat = 50

plt.plot(r_times[0:-1] - d_times)
#plt.plot(np.diff(r_times[0:lookat]))
#plt.plot(np.diff(d_times[0:lookat]))
plt.show()


"""
rgb = open(rgb_path)
depth = open(depth_path)

rgb.next()
rgb.next()
rgb.next()

depth.next()
depth.next()
depth.next()

rgbs = []
for r in rgb:
    rgbs.appen(float(r.split(' ')[0]))
rgbs = np.array(rgbs)
print rgbs
"""

