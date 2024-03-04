import gradio as gr
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def visualize_clusters(dataset_file):
  """Performs K-Means clustering on the provided dataset and visualizes the clusters."""

  # Read the CSV file using pandas, handling potential errors gracefully
  try:
    dataset = pd.read_csv(dataset_file)
  except pd.errors.ParserError as e:
    print(f"Error parsing CSV: {e}")
    return None  # Indicate an error occurred

  X = dataset.iloc[:, [3, 4]].values  # Extract relevant features

  # Perform K-Means clustering with 5 clusters
  kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
  y_kmeans = kmeans.fit_predict(X)

  # Create a visually appealing plot
  plt.figure(figsize=(8, 6))  # Adjust plot size for better visibility
  colors = ['red', 'blue', 'green', 'cyan', 'magenta']  # Define a color list for clusters
  for i in range(5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')

  # Emphasize centroids with larger markers and a distinct color
  plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='gold', marker='*', label='Centroids')

  # Add informative labels and title
  plt.title('K-Means Clustering of Customers')
  plt.xlabel('Annual Income (k$)')
  plt.ylabel('Spending Score (1-100)')
  plt.legend()

  # Render the plot as a PNG image for Gradio
  plt.tight_layout()
  plt.savefig("clusters.png")  # Save plot to a temporary file

  return gr.Image("clusters.png")

# Create the Gradio interface with informative descriptions
iface = gr.Interface(
  visualize_clusters,
  inputs=[gr.File(label="Upload your CSV dataset (expected format: annual income, spending score)", type="filepath")],
  outputs=gr.Image(label="Cluster Visualization"),
  title="K-Means Clustering with Gradio",
  description="Visualize clusters in your CSV dataset containing annual income and spending score columns."
)

iface.launch()