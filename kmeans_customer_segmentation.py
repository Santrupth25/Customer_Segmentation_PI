import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Display first rows
print(data.head())

# Select features for clustering
# Using Annual Income and Spending Score
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Use Elbow Method to find optimal clusters
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure()
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply K-Means with optimal clusters (usually 5)
kmeans = KMeans(n_clusters=5, random_state=42)

y_kmeans = kmeans.fit_predict(X)

# Add cluster labels to dataset
data['Cluster'] = y_kmeans

print(data.head())

# Visualize clusters
plt.figure()

plt.scatter(X[y_kmeans == 0]['Annual Income (k$)'],
            X[y_kmeans == 0]['Spending Score (1-100)'],
            s=100,
            label='Cluster 1')

plt.scatter(X[y_kmeans == 1]['Annual Income (k$)'],
            X[y_kmeans == 1]['Spending Score (1-100)'],
            s=100,
            label='Cluster 2')

plt.scatter(X[y_kmeans == 2]['Annual Income (k$)'],
            X[y_kmeans == 2]['Spending Score (1-100)'],
            s=100,
            label='Cluster 3')

plt.scatter(X[y_kmeans == 3]['Annual Income (k$)'],
            X[y_kmeans == 3]['Spending Score (1-100)'],
            s=100,
            label='Cluster 4')

plt.scatter(X[y_kmeans == 4]['Annual Income (k$)'],
            X[y_kmeans == 4]['Spending Score (1-100)'],
            s=100,
            label='Cluster 5')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            s=300,
            marker='X',
            label='Centroids')

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()