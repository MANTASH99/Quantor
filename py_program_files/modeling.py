import numpy as np 
import pandas as pd 
import matplotlib
matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



# def convert_first_point_to_decimal(val):
#     if isinstance(val, str):  # Check if the value is a string
#         parts = val.split('.')  # Split the string by periods
#         if len(parts) > 1:
#             return float(parts[0] + '.' + ''.join(parts[1:]))  # Use the first part as the integer and combine others as decimals
#         elif len(parts) == 1:
#             return float(parts[0])  # Handle strings without any periods
#     return val

data = pd.read_csv('updated_file_ICEPHI_summed.csv', delimiter=',')

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
    'privat': (0, 102),
    'public': (103, 187),
    'Unscripted': (188, 406),
    'Scripted': (407, 491),
    'Student Writing': (492, 539),
    'Letters': (540, 716),
    'Academic Writing': (717, 758),
    'Popular Writing': (759, 836),
    'Reportage': (837, 923),
    'Instructional Writing': (924, 961),
    'persuasive writing': (962, 1003),
    'Creative Writing': (1004, None)
}

colors = []
labels = []

# Assign colors based on ranges
for i in range(len(pca_result)):
    for label, (start, end) in class_ranges.items():
        if end is None and i >= start - 1:
            colors.append('orange')  # Default color for undefined range
            labels.append(label)
            break
        elif start - 1 <= i < end:
            color_map = {
                'privat': 'red',
                'public': 'blue',
                'Unscripted': 'green',
                'Scripted': 'purple',
                'Student Writing': 'yellow',
                'Letters': 'pink',
                'Academic Writing': 'brown',
                'Popular Writing': 'cyan',
                'Reportage': 'magenta',
                'Instructional Writing': 'lime',
                'persuasive writing': 'teal'
            }
            colors.append(color_map.get(label, 'black'))  # Default to black if label not in color_map
            labels.append(label)
            break

label_to_num = {label: idx for idx, label in enumerate(class_ranges.keys())}
numeric_labels = [label_to_num[label] for label in labels]

lda = LDA(n_components=3)  # Choose the number of components (max: n_classes - 1)
lda_result = lda.fit_transform(pca_result, numeric_labels)


plt.figure(figsize=(8, 6))
plt.scatter(lda_result[:, 0], lda_result[:, 1], c=colors, edgecolor='k', s=50)
plt.title('LDA - 2 Components')
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.grid()

# Add legend to the 2D plot
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) 
           for label, color in zip(class_ranges.keys(), 
                                   ['red', 'blue', 'green', 'purple', 'yellow', 
                                    'pink', 'brown', 'cyan', 'magenta', 'lime', 'teal', 'orange'])]
plt.legend(handles=handles, title="Classes", loc="best")

# Save the 2D LDA plot
plt.savefig('lda_2d_plot_PHI.png')
print("2D LDA plot saved as 'lda_2d_plot.png'.")

# Visualize LDA results in 3D (if 3 components are available)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(lda_result[:, 0], lda_result[:, 1], lda_result[:, 2], c=colors, edgecolor='k', s=50)
ax.set_title('LDA - 3 Components')
ax.set_xlabel('Linear Discriminant 1')
ax.set_ylabel('Linear Discriminant 2')
ax.set_zlabel('Linear Discriminant 3')

# Add legend to the 3D plot
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) 
           for label, color in zip(class_ranges.keys(), 
                                   ['red', 'blue', 'green', 'purple', 'yellow', 
                                    'pink', 'brown', 'cyan', 'magenta', 'lime', 'teal', 'orange'])]
ax.legend(handles=handles, title="Classes", loc="best")

# Save the 3D LDA plot
plt.savefig('lda_3d_plot_PHI.png')
print("3D LDA plot saved as 'lda_3d_plot.png'.")

# Optional: Interactive LDA Visualization using Plotly
import plotly.express as px
lda_df = pd.DataFrame(lda_result, columns=['LD1', 'LD2', 'LD3'])
lda_df['Color'] = labels

fig = px.scatter_3d(lda_df, x='LD1', y='LD2', z='LD3', color='Color', 
                    title="3D LDA Plot", 
                    labels={'LD1': 'Linear Discriminant 1', 'LD2': 'Linear Discriminant 2', 'LD3': 'Linear Discriminant 3'})

# Save the interactive plot as an HTML file
fig.write_html("lda_3d_interactive_plot_PHI.html")
print("3D LDA interactive plot saved as 'lda_3d_interactive_plot.html'.")

# Show the interactive plot
fig.show()

# Explained variance ratio for LDA components
lda_explained_variance = lda.explained_variance_ratio_
print(f"Explained variance by each LDA component: {lda_explained_variance}")




# 2D PCA Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, edgecolor='k', s=50)
plt.title('PCA - 2 Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()

# Add legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) 
           for label, color in zip(class_ranges.keys(), ['red', 'blue', 'green', 'purple', 'yellow', 
                                                        'pink', 'brown', 'cyan', 'magenta', 'lime', 'teal', 'orange'])]
plt.legend(handles=handles, title="Classes", loc="best")

# Save and display the plot
plt.savefig('pca_2d_plot_PHI.png')
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
fig.savefig('pca_3d_plot_PHI.png')  # Save the 3D plot
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
fig.write_html("pca_3d_interactive_plot_PHI.html")