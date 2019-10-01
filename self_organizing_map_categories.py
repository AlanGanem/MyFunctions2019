# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\ganem\Desktop\2Vintens\CSVs exportados\ranking(01-16-01.2019).csv', encoding = 'ANSI')
dataset= dataset[['demand_by_supply','ganem_x','amount_of_sellers','paused_over_total','revenues_by_supply','category_id','ganem_y','gini_coefficient_sold','average_price']]
A_categories_ = dataset[dataset.category_id.isin(A_categories)]
A_categories_X = A_categories_.iloc[:, :-1].values
A_categories_y = A_categories_.iloc[:,-1].values
X = dataset.iloc[:, :-1].values
y = dataset['category_id']
indexes = A_categories_.index
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)
A_categories_X = sc.fit_transform(A_categories_X)

# Training the SOM
plt.clf()
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 8, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)
som.distance_map()
win_map = som.win_map(A_categories_X)
(4,9) 

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

response = som.activation_response(A_categories_X[1])
som.winner(A_categories_X[2])
for i, x in enumerate(A_categories_X):
    w = som.winner(A_categories_X[i])
    plot(w[0] + np.random.random(),
         w[1] + np.random.random(),
         'o',
         markersize = 1,
         markeredgewidth = 2)
show()

list_of_cats = [y[i] for i in range(len(X)) if (som.winner(np.array(X[i]))) in win_map.keys()]

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)