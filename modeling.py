import numpy as np 
import pandas as pd 
import matplotlib
matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
import plotly.express as px




# def convert_first_point_to_decimal(val):
#     if isinstance(val, str):  # Check if the value is a string
#         parts = val.split('.')  # Split the string by periods
#         if len(parts) > 1:
#             return float(parts[0] + '.' + ''.join(parts[1:]))  # Use the first part as the integer and combine others as decimals
#         elif len(parts) == 1:
#             return float(parts[0])  # Handle strings without any periods
#     return val
data = pd.read_csv('Summed_new_features.csv', delimiter=',')

new_data = data.drop('group', axis=1)
# for col in data.columns:
#     data[col] = data[col].apply(convert_first_point_to_decimal)

print(data.head())


# new_data = data.drop('id', axis=1)
data_ready= np.array(new_data.values)

print(data_ready)
print(data.dtypes)
# Check for NaN values


# Check for infinite values



scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_ready)

pca= PCA(n_components= 3)

pca_result =pca.fit_transform(data_scaled)

explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by each component: {explained_variance}")


loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(pca.n_components_)], index=new_data.columns)

# Print the loadings for each principal component
print("\nLoadings for each principal component (PC1, PC2, PC3):")
print(loadings)

# Identify the top features with the highest absolute weights for each component
top_pc1 = loadings['PC1'].abs().sort_values(ascending=False).head(10)
top_pc2 = loadings['PC2'].abs().sort_values(ascending=False).head(10)
top_pc3 = loadings['PC3'].abs().sort_values(ascending=False).head(10)

print("\nTop 10 features with the highest absolute loadings for PC1:")
print(top_pc1)

print("\nTop 10 features with the highest absolute loadings for PC2:")
print(top_pc2)

print("\nTop 10 features with the highest absolute loadings for PC3:")
print(top_pc3)
n_points = pca_result.shape[0]  # Total number of data points
colors = []
class_ranges = {
    'Public': (1, 100),
    'Creative Writing': (101, 177),
    'Unscripted': (178, 246),
    'Scripted': (247, 296),
    'Student Writing': (297, 316),
    'Letters': (317, 346),
    'Academic Writing': (347, 386),
    'Popular Writing': (387, 426),
    'Reportage': (427, 446),
    'Instructional Writing 1': (447, 466),
    'Instructional Writing 2': (467, 476),
    'Creative Writing (Others)': (477, None),  # For the rest of the points
}

# Map each index to its corresponding class and color
colors = []
labels = []

for i in range(len(pca_result)):
    for label, (start, end) in class_ranges.items():
        if end is None:
            if i >= start - 1:  # For the rest after the last defined range
                colors.append('orange')  # Color for remaining class
                labels.append(label)
                break
        else:
            if start - 1 <= i < end:
                if label == 'Public':
                    colors.append('red')
                elif label == 'Creative Writing':
                    colors.append('blue')
                elif label == 'Unscripted':
                    colors.append('green')
                elif label == 'Scripted':
                    colors.append('purple')
                elif label == 'Student Writing':
                    colors.append('yellow')
                elif label == 'Letters':
                    colors.append('pink')
                elif label == 'Academic Writing':
                    colors.append('brown')
                elif label == 'Popular Writing':
                    colors.append('cyan')
                elif label == 'Reportage':
                    colors.append('magenta')
                elif label == 'Instructional Writing 1':
                    colors.append('lime')
                elif label == 'Instructional Writing 2':
                    colors.append('teal')
                labels.append(label)
                break

# 2D Visualization
plt.figure(figsize=(8, 6))

# Scatter plot with color labels
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, edgecolor='k', s=50)
plt.title('PCA - 2 Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()

# Add a legend manually
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) 
           for color, label in zip(
               ['red', 'blue', 'green', 'purple', 'yellow', 'pink', 'brown', 'cyan', 'magenta', 'lime', 'teal'],
               class_ranges.keys())]
plt.legend(handles=handles, title="Classes")

# Save the 2D plot
plt.savefig('pca_2d_plot.png')  # Save the 2D plot
print("2D PCA plot saved as 'pca_2d_plot.png'.")

# For 3D Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for 3D
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=colors, edgecolor='k', s=50)
ax.set_title('PCA - 3 Components')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

# Add a legend manually
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) 
           for color, label in zip(
               ['red', 'blue', 'green', 'purple', 'yellow', 'pink', 'brown', 'cyan', 'magenta', 'lime', 'teal'],
               class_ranges.keys())]
ax.legend(handles=handles, title="Classes")

# Save the 3D plot
fig.savefig('pca_3d_plot.png')  # Save the 3D plot
print("3D PCA plot saved as 'pca_3d_plot.png'.")

# Show the plot (optional, comment this if you are running in a non-interactive environment)
plt.show()

# Prepare the DataFrame for Plotly (Interactive Plot)
df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
df['Color'] = labels

# Create a 3D scatter plot using Plotly
fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Color', 
                    title="3D PCA Plot", 
                    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'})

# Show the interactive plot
fig.show()

# Optionally save the interactive plot as an HTML file
fig.write_html("pca_3d_interactive_plot.html")