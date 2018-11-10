# All Code from https://github.com/llSourcell/visualize_dataset_demo
# Siraj Raval

# Necessary Libraries for T-SNE Visualization
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Download the Data Set
dataframe_all = pd.read_csv("mfcc_viz.csv")

# Extract and Scale the Features into a Numpy Array
x = dataframe_all.ix[:,:-1].values
standard_scaler = StandardScaler()
x_std = standard_scaler.fit_transform(x)

# Retrieve Class Labels (real or fake)
y = dataframe_all.ix[:,-1].values

# Encode the Class Label
class_labels = np.unique(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split Data into Training and Test Split
test_percentage = 0.1
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = test_percentage, random_state = 0)

# t-distributed Stochastic Neighbor Embedding (t-SNE) Visualization
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
x_test_2d = tsne.fit_transform(x_test)

# Plot the Distribution of the Two Classes
markers=('s', 'd')
color_map = {0:'red', 1:'blue'}
plt.figure()
for idx, cl in enumerate(np.unique(y_test)):
    plt.scatter(x=x_test_2d[y_test==cl,0], y=x_test_2d[y_test==cl,1], c=color_map[idx], marker=markers[idx], label=cl)

# Add Legend, Title, and Labels
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper right', fontsize=25)
plt.title('t-SNE Visualization of Test Data', fontsize=15)

# View Generated Plot
plt.show()
