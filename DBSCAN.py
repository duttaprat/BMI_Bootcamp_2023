#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics,datasets
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


# In[2]:


# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)


# In[3]:


X = StandardScaler().fit_transform(X)


# In[8]:


db = DBSCAN(eps=0.4, min_samples=2).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)


core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print (set(labels))


# In[9]:


n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print (n_clusters_)


# In[10]:


import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




