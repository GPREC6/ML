import pandas as pd
import numpy as np

data = pd.read_csv('iris - iris.csv')
data.head()

x = data.iloc[:,[0,1,2,3]].values
x

from sklearn.cluster import KMeans
wcss = []  
for i in range(1,16):  
    kmeans = KMeans(n_clusters=i,random_state=1)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

wcss 

import matplotlib.pyplot as plt
plt.plot(range(1,16),wcss)
plt.title("Elbow Method")
plt.xlabel("No. of clusters")
plt.ylabel("WCSS")

kmeans = KMeans(n_clusters=3,random_state=1)
y_predict = kmeans.fit_predict(x)
y_predict

kmeans1 = KMeans(n_clusters=4,random_state=1)
y_predict1 = kmeans1.fit_predict(x)
y_predict1

from sklearn.metrics import silhouette_score


sil_score = silhouette_score(x,kmeans.labels_)
sil_score


sil_score1 = silhouette_score(x,kmeans1.labels_)
sil_score1

plt.scatter(x[y_predict == 0,0], x[y_predict == 0,1],c='red',label='First Cluster')
plt.scatter(x[y_predict == 1,0], x[y_predict == 1,1],c='blue',label='Second  Cluster')
plt.scatter(x[y_predict == 2,0], x[y_predict == 2,1],c='green',label='Third Cluster')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='yellow',label='Centroid')
plt.xlabel('Sepal_Length')
plt.ylabel('Sepal_Width')


plt.scatter(x[y_predict == 0,2], x[y_predict == 0,3],c='red',label='First Cluster')
plt.scatter(x[y_predict == 1,2], x[y_predict == 1,3],c='blue',label='Second  Cluster')
plt.scatter(x[y_predict == 2,2], x[y_predict == 2,3],c='green',label='Third Cluster')
plt.scatter(kmeans.cluster_centers_[:,2],kmeans.cluster_centers_[:,3],c='yellow',label='Centroid')
plt.xlabel('Petal_Length')
plt.ylabel('Petal_Width')
plt.legend()