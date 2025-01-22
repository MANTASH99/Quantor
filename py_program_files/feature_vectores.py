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
########################       input     output      STD    ##################


# def convert_first_point_to_decimal(val):
#     if isinstance(val, str):  # Check if the value is a string
#         parts = val.split('.')  # Split the string by periods
#         if len(parts) > 1:
#             return float(parts[0] + '.' + ''.join(parts[1:]))  # Use the first part as the integer and combine others as decimals
#         elif len(parts) == 1:
#             return float(parts[0])  # Handle strings without any periods
#     return val

data = pd.read_csv('summed_document_generalized.csv', delimiter=',')

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
print('input' ,data_ready.shape,data_ready)


print('output', data_scaled.shape, data_scaled)

