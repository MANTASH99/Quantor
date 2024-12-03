import numpy as np 
import pandas as pd 
import matplotlib
matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
import plotly.express as px

data = pd.read_csv('Summed.csv', delimiter=',')
print(data.head())
new_data = data.drop('id', axis=1)
new_data = new_data.drop('group', axis=1)
data_ready= np.array(new_data.values)


scaler = StandardScaler()

data_scaled= scaler.fit_transform(data_ready)

print(data_scaled)

pca= PCA(n_components= 3)

pca_result =pca.fit_transform(data_scaled)

explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by each component: {explained_variance}")

n_points = pca_result.shape[0]  # Total number of data points
colors = []
for i in range(n_points):
    if i < 177:
        colors.append('red')          # First 177 points
    elif i < 296:
        colors.append('blue')         # Next 119 points (177 to 295)
    elif i < 346:
        colors.append('green')        # Next 50 points (296 to 345)
    elif i < 496:
        colors.append('purple')       # Next 150 points (346 to 495)
    else:
        colors.append('orange') 



# 2D Visualization
plt.figure(figsize=(8, 6))

#Scatter plot with color labels
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, edgecolor='k', s=50)
plt.title('PCA - 2 Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()

# Save the 2D plot
plt.savefig('pca_2d_plot.png')  # Save the 2D plot
print("2D PCA plot saved as 'pca_2d_plot.png'.")

# For 3D Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
pca_3d = PCA(n_components=3)
pca_result_3d = pca_3d.fit_transform(data_scaled)

# Scatter plot for 3D
ax.scatter(pca_result_3d[:, 0], pca_result_3d[:, 1], pca_result_3d[:, 2], c=colors, edgecolor='k', s=50)
ax.set_title('PCA - 3 Components')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

# Save the 3D plot
fig.savefig('pca_3d_plot.png')  # Save the 3D plot
print("3D PCA plot saved as 'pca_3d_plot.png'.")

# Show the plot (optional, comment this if you are running in a non-interactive environment)
plt.show()

df = pd.DataFrame(pca_result_3d, columns=['PC1', 'PC2', 'PC3'])

# Color the points: First 296 points in red, the rest in blue
n_points = df.shape[0]
colors = ['red' if i < 296 else 'blue' for i in range(n_points)]
df['Color'] = colors

# Create a 3D scatter plot
fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Color', 
                    title="3D PCA Plot", 
                    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'})

# Show the plot
fig.show()

# Optionally save the interactive plot as an HTML file
fig.write_html("pca_3d_interactive_plot.html")
