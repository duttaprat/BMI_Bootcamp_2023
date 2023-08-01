#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import matplotlib.pyplot as plot
from matplotlib.pyplot import style
style.use("seaborn-darkgrid")


# # Importing Kmeans function from sklearn package

# In[33]:


from sklearn.cluster import KMeans


# # Importing silhouette score which act as a cluster validity index for unlabeled data

# In[34]:


from sklearn.metrics import silhouette_score


# # Importing adjusted_rand_score which act as a cluster validity index for labeled data

# In[35]:


from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


# # Load the 2D data which is a user input 

# In[36]:


all_data = np.array([[1, 2], [5,8], [1.5, 1.8] ,[8,8], [9,11], [10,1], [7,5], [9,2], [3,7]])
a = all_data
print (a)


# 
# 
# # Call the Kmeans function and train the data

# In[43]:


kmeans = KMeans(n_clusters=2)
kmeans.fit(a)


# # Obtain centroids and labels as parameters

# In[44]:


centroids = kmeans.cluster_centers_
label = kmeans.labels_
print ("Cluster Centers are :", centroids)
print ("Labels :", label)


# # Plotting the data and their clusters

# In[45]:


colours = ['m.','g.','r.', 'k.']
for i in range(len(a)):
    #print "Coordinates ", a[i], "labels", label[i]
    plot.plot(a[i][0], a[i][1], colours[label[i]], markersize = 10 )
for i in range(len(centroids)):
    plot.plot(centroids[i][0], centroids[i][1], colours[i], markersize = 12, marker='*' )
plot.show()


# # Calculate the cluster goodness

# In[46]:


Sil_score=silhouette_score(a,label)
print ("Silhouette Score: ", Sil_score)


# # Now we will load unlabeled data from file. Preprocessing the data from the file is the main task. All the remaining part is same as previous method.

# In[47]:


read_file=open("Data_set/21D_data.txt",'r')
read_content= read_file.read()


# # Getting number of data points and number of features/samples for each data point

# In[49]:


all_data= read_content.splitlines()
No_data_points=len(all_data)
print ("Number of the data points :- ", No_data_points)
features = all_data[0].split("\t")
No_of_features = len(features)
print ("Number the features/samples :- ", No_of_features)


# # Loading the whole dataset in a 2D matrix

# In[50]:


a=np.zeros((No_data_points,No_of_features))
counter = 0
for lines in all_data:
    values=lines.split('\t')
    for i in range(0,No_of_features):
        a[counter][i]= values[i]
    counter+=1
    
print (a)    # If you want to see the whole dataset


# # Call the Kmeans function and train the data

# In[51]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(a)


# # Obtain centroids and labels as parameters

# In[52]:


centroids = kmeans.cluster_centers_
label = kmeans.labels_
print ("Cluster Centers are :", centroids)
print ("Labels :", label)


# # Plotting the data and their clusters

# In[25]:


colours = ['m.','g.','r.', 'k.']
for i in range(len(a)):
    #print "Coordinates ", a[i], "labels", label[i]
    plot.plot(a[i][0], a[i][1], colours[label[i]], markersize = 10 )
for i in range(len(centroids)):
    plot.plot(centroids[i][0], centroids[i][1], colours[i], markersize = 12, marker='*' )
plot.show()


# # Now we will load labeled data from file. Preprocessing the data from the file is the main task. All the remaining part is same as previous method.

# In[53]:


read_file=open("Data_set/iris_org.txt",'r')
read_content= read_file.read()


# # Getting number of data points and number of features/samples for each data point. 

# In[54]:


all_data= read_content.splitlines()
No_data_points=len(all_data)
print ("Number of the data points :- ", No_data_points)
features = all_data[0].split("\t")
No_of_features = len(features)-1
print ("Number the features/samples :- ", No_of_features)


# # Loading the whole dataset in a 2D matrix. Also get the true labels

# In[55]:


a=np.zeros((No_data_points,No_of_features))
true_label = []
counter = 0
for lines in all_data:
    values=lines.split('\t')
    for i in range(0,No_of_features):
        a[counter][i]= values[i]
    true_label.append(int(values[No_of_features]))     #
    counter+=1


# In[30]:


kmeans = KMeans(n_clusters=5)
kmeans.fit(a)
centroids = kmeans.cluster_centers_
label = kmeans.labels_
print ("Cluster Centers are :", centroids)
print ("Labels :", label)


# In[31]:


colours = ['m.','g.','r.', 'k.','b.']
for i in range(len(a)):
    plot.plot(a[i][0], a[i][1], colours[label[i]], markersize = 10 )
for i in range(len(centroids)):
    plot.plot(centroids[i][0], centroids[i][1], colours[i], markersize = 12, marker='*' )
plot.show()


# In[ ]:





# In[ ]:





# In[ ]:




