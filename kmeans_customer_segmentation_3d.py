import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

print(data.head())

# Select 3 features for 3D clustering
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)

# Add cluster column
data['Cluster'] = clusters

# 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    data['Age'],
    data['Annual Income (k$)'],
    data['Spending Score (1-100)'],
    c=data['Cluster']
)

# Plot centroids
centroids = kmeans.cluster_centers_

ax.scatter(
    centroids[:,0],
    centroids[:,1],
    centroids[:,2],
    s=300,
    marker='X'
)

ax.set_title("3D Customer Segmentation using K-Means")
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score")

plt.show()