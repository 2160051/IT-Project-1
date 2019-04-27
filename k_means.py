# %matplotlib inline
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
#feature selection
from sklearn.ensemble import ExtraTreesClassifier
#silhoutte coefficient
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets

#Load dataset
link = 'https://raw.githubusercontent.com/2160051/IT-Project-1/master/Data%20Analytics.csv'
plotdata = pd.read_csv(link)

#Multiple Attributes - multiple variables
xattributes = plotdata[['Population_Density(km²)', 'TAVE', 'Glaciers', 'Locales', 'Beaches', 'Areas', 'Lakes', 'Streams', 'Swamps', 'Forests', 'Plains', 'Woods']]
xattributes = xattributes[:50] #only using the first 50 rows
xattributes = np.array(xattributes)

#X
xdata = plotdata[['Population_Density(km²)']] #used population density as the sample
xdata = xdata[:50]
xdata = np.array(xdata['Population_Density(km²)'].values)

#Y
ydata = plotdata[['Person-to-Person Contact']]
ydata = ydata[:50] #only using the first 50 rows
ydata = np.array(ydata['Person-to-Person Contact'].values)

#2 attribute plot
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
scatterplot = np.array(list(zip(xdata, ydata)))
kmeans = KMeans(n_clusters=4) #low, medium, high, very high
kmeans = kmeans.fit(scatterplot)
datacentroids = kmeans.cluster_centers_
print("Cluster Centroids:")
print(datacentroids.tolist()) 
plt.scatter(scatterplot[:, 0], scatterplot[:, 1], c=kmeans.labels_, cmap='rainbow') 
plt.scatter(datacentroids[:, 0], datacentroids[:, 1], c='black', s=100, alpha=0.5)  
plt.xlabel('Population_Density(km²)')
plt.ylabel('Total Number of Cases Categorized as Person-to-Person Contact')
plt.title('Scatter Plot with K-means Clusters')
plt.show()

#Silhouette coefficient
print("Silhouette Coefficient:")
kmeans_model = KMeans(n_clusters=4, random_state=1).fit(scatterplot)
labels = kmeans_model.labels_
metrics.silhouette_score(scatterplot, labels, metric='euclidean')

#3 attribute plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#X
xt = plotdata[['Population_Density(km²)']] #used population density as the sample
xt = xt[:50]
xt = np.array(xt['Population_Density(km²)'].values)

#Y
yt = plotdata[['Person-to-Person Contact']]
yt = yt[:50] #only using the first 50 rows
yt = np.array(yt['Person-to-Person Contact'].values)

#Z
zt = plotdata[['GDP per capita']]
zt = zt[:50] #only using the first 50 rows
zt = np.array(zt['GDP per capita'].values)

ax.scatter(xt, yt, zt)
ax.set_xlabel('Population_Density')
ax.set_ylabel('Number of Cases')
ax.set_zlabel('GDP per Capita')
ax.axis('tight')

tscatterplot = np.array(list(zip(xt, yt, zt)))
kmeans = KMeans(n_clusters=4) #low, medium, high, very high
kmeans = kmeans.fit(tscatterplot)
tdatacentroids = kmeans.cluster_centers_
print("Cluster Centroids:")
print(tdatacentroids.tolist()) 
ax.scatter(tscatterplot[:, 0], tscatterplot[:, 1], tscatterplot[:, 2], c=kmeans.labels_, cmap='rainbow') 
ax.scatter(tdatacentroids[:, 0], tdatacentroids[:, 1], tdatacentroids[:, 2], c='black', s=50, alpha=0.5)  
plt.title('Number of cases, GDP and population density')
plt.show()

#K-means table
clusters = 4
df = plotdata.copy(deep=True)
firstcolumn = df.columns[0] #index column
df.drop([firstcolumn], axis = 1, inplace = True)
df = df.drop('State', 1)
df = df.drop('Week', 1)
km = KMeans(n_clusters=clusters, random_state=1)
new = df._get_numeric_data()
km.fit(new)
predict=km.predict(new)
centroids = km.cluster_centers_
dfkmeans = df.copy(deep=True)
dfkmeans['KMeans Cluster'] = pd.Series(predict, index=dfkmeans.index)
kmeanstable = plotdata.copy(deep=True)
kmeanstable['KMeans Cluster'] = pd.Series(predict, index=dfkmeans.index)
indexcolumn = kmeanstable.columns[0]
kmeanstable.drop(indexcolumn, axis = 1, inplace = True)
print(kmeanstable.to_string())
print(centroids)

#Group rows based on number of clusters
for i in range(0, clusters):
  use = kmeanstable[kmeanstable['KMeans Cluster'] == i]
  clusterx = use['Population_Density(km²)']
  clustery = use['Person-to-Person Contact']
  #graph cluster attributes
  plt.rcParams['figure.figsize'] = (16, 9)
  plt.style.use('ggplot')
  clusterscatterplot = np.array(list(zip(clusterx, clustery)))
  kmeans = KMeans(n_clusters=4) #low, medium, high, very high
  kmeans = kmeans.fit(clusterscatterplot)
  datacentroids = kmeans.cluster_centers_
  print("Cluster "+str(i))
  print("Cluster Centroids:")
  print(datacentroids.tolist()) 
  plt.scatter(clusterscatterplot[:, 0], clusterscatterplot[:, 1], c=kmeans.labels_, cmap='rainbow') 
  plt.scatter(datacentroids[:, 0], datacentroids[:, 1], c='black', s=100, alpha=0.5)  
  plt.xlabel('Population_Density(km²)')
  plt.ylabel('Total Number of Cases Categorized as Person-to-Person Contact')
  plt.title('Scatter Plot with KMeans Clusters')
  plt.show()
  i+=1
