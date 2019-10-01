# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:32:21 2019

@author: PC10
"""
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler
from keras.layers import Input, Dense, Dropout
from keras.models import Model
list(ranking)
#get the best candidates of relevant features
removed_features = ['product_name','product_id','product_id_by_price','top_sellers','top_ads','father_category']
label_encode =False
features = [i for i in ranking.columns if i not in removed_features]
ranking_filtered = ranking.loc[:,features]
X = ranking_filtered.values
#one hot encode category_name 

# scale features
if label_encode:
    label_encoder = LabelEncoder()
    X[:,1] = label_encoder.fit_transform(X[:,1])
    
    one_hot_encoder = OneHotEncoder(categorical_features= [1], sparse = False, n_values = 'auto')
    X = one_hot_encoder.fit_transform(X[:,0:34])

standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X[:,0:X.shape[1]-1])
#min_max_scaler = MinMaxScaler()
#X = min_max_scaler.fit_transform(X)

from sklearn.decomposition import PCA,KernelPCA
pca = PCA(n_components = 15)
pca.fit(X)
explained_variance = pca.explained_variance_ratio_
explained_variance.sum()

X= pca.fit_transform(X)


'''X =  np.reshape(X,(X.shape[0],X.shape[1]))
## model
f1 = 1
f2 = 0.5

encoder_input = Input(shape = (X.shape[1],))

encoder_in = Dense(int(f1*X.shape[1]), name = 'encoder_in')(encoder_input)
d1 = Dropout(0.2)(encoder_in)
encoder_out = Dense(int(f2*X.shape[1]), name = 'encoder_1')(d1)
d2 = Dropout(0.2)(encoder_out)

#decoder_1 = Dense(int(f1*X.shape[1]), name = 'decoder_1')(d2)
#d4 = Dropout(0.2)(decoder_1)
output = Dense(int(X.shape[1]))(d2)

autoencoder = Model(encoder_input, output)
encoder = Model (encoder_input, encoder_out)

from keras.utils import plot_model
plot_model(autoencoder,show_shapes = True, to_file='model.png')

autoencoder.compile(optimizer = 'adam',loss = 'mse')

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

history = autoencoder.fit(x = X, y = X,batch_size = 1000, epochs = 100,validation_split = 0.2)#,callbacks = [es]

plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

autoencoder.predict(X[1,:].reshape(1,-1))
encoder.predict(X[1,:].reshape(1,-1))

X_encoded = encoder.predict(X)
X_encoded = X_encoded.dot(X_encoded.T)'''

import fastcluster
import  scipy
linkage = fastcluster.linkage_vector(X, method ='ward')
labels = scipy.cluster.hierarchy.fcluster(linkage,t = 0.1, criterion='distance')
len(np.unique(labels))
ranking= ranking.assign(cluster_labels = labels)


label = ranking[ranking['product_id'] == 20544]['cluster_labels'].max()
view = ranking[ranking['cluster_labels'].between(label-10,label+10)]