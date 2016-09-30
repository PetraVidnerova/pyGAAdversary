import pandas as pd 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA



#load csv 
data = pd.read_csv("models.csv")

X = data[data.columns[:-1]]
targets = data[data.columns[-1]]

X_reduced = PCA(n_components=2).fit_transform(X)

print(X_reduced)


fig = plt.figure(1, figsize=(8, 6))

plt.scatter(X_reduced[:,0], X_reduced[:,1])

for label, x, y in zip(targets, X_reduced[:, 0], X_reduced[:, 1]):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (-10, 10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
    )



#ax = Axes3D(fig, elev=-150, azim=110)

#ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2])
#ax.set_title("First three PCA directions")
#ax.set_xlabel("1st eigenvector")
#ax.w_xaxis.set_ticklabels([])
#ax.set_ylabel("2nd eigenvector")
#ax.w_yaxis.set_ticklabels([])
#ax.set_zlabel("3rd eigenvector")
#ax.w_zaxis.set_ticklabels([])

plt.show()
