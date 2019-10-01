
from keras.layers import Dense
from keras.models import Sequential
import pickle
from func_search_engine import search_engine
import os
import pandas as pd
from sklearn.decomposition import KernelPCA, PCA
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler


janeiro_abc  = pd.read_csv(r'C:\ProductClustering\input_Data\Janeiropareto.csv',encoding= 'ANSI')
ranking_janeiro = pd.read_csv(r'C:\ProductClustering\output_data\csv_files\product_rankings\2019-01-01_to_2019-01-31.csv',encoding = 'ANSI')
clustered_data_janeiro = pd.read_csv(r'C:\ProductClustering\output_data\csv_files\clustered_ads\2019-01-01_to_2019-01-31.csv')
g = os.path.join(os.path.dirname(r'C:\ProductClustering\output_data\\'), 'cv_matrix')
cv_matrix = pickle.load(open(g, 'rb'))
g = os.path.join(os.path.dirname(r'C:\ProductClustering\output_data\\'), 'vocabulary')
vocabulary = pickle.load(open(g, 'rb'))

janeiro_a = janeiro_abc[janeiro_abc.Categoria == 'A']
janeiro_b = janeiro_abc[janeiro_abc.Categoria == 'B']
janeiro_c = janeiro_abc[janeiro_abc.Categoria == 'C']

ids_a =[]
for ad in janeiro_a['Row Labels']:
    a = search_engine(ad,cv_matrix,vocabulary, metric = 'cosine')    
    ids_a.append(clustered_data_janeiro['product_id'].iloc[a[0][0]])
janeiro_a = janeiro_a.assign(product_id = ids_a)
janeiro_a =janeiro_a[['product_id','Sum of Revenue']]  
janeiro_a = janeiro_a.groupby('product_id').sum().reset_index()


ids_b =[]
for ad in janeiro_b.iloc[:,0] :
    a = search_engine(ad,cv_matrix,vocabulary, metric = 'cosine')    
    ids_b.append(clustered_data_janeiro['product_id'].iloc[a[0][0]])
janeiro_b = janeiro_b.assign(product_id = ids_b)
janeiro_b =janeiro_b[['product_id','Sum of Revenue']]  
janeiro_b = janeiro_b.groupby('product_id').sum().reset_index()

ids_c =[]
for ad in janeiro_c.iloc[:,0] :
    a = search_engine(ad,cv_matrix,vocabulary, metric = 'cosine')    
    ids_c.append(clustered_data_janeiro['product_id'].iloc[a[0][0]])
janeiro_c = janeiro_c.assign(product_id = ids_c)
janeiro_c = janeiro_c[['product_id','Sum of Revenue']]  
janeiro_c = janeiro_c.groupby('product_id').sum().reset_index()

ads_a = ranking_janeiro[ranking_janeiro.product_id.isin(ids_a)].iloc[:,~ranking_janeiro.columns.isin(['title_corpus','product_name','Unnamed: 0','product_id','top_sellers','relative_price_range','price_range','top_ads'])]
ads_b = ranking_janeiro[ranking_janeiro.product_id.isin(ids_b)].iloc[:,~ranking_janeiro.columns.isin(['title_corpus','product_name','Unnamed: 0','product_id','top_sellers','relative_price_range','price_range','top_ads'])]
ads_c = ranking_janeiro[ranking_janeiro.product_id.isin(ids_c)].iloc[:,~ranking_janeiro.columns.isin(['title_corpus','product_name','Unnamed: 0','product_id','top_sellers','relative_price_range','price_range','top_ads'])]

ads_a = ads_a.assign(revenues = janeiro_a['Sum of Revenue'].values)
ads_b = ads_b.assign(revenues = janeiro_b['Sum of Revenue'].values)
ads_c = ads_c.assign(revenues = janeiro_c['Sum of Revenue'].values)

all_ads = ads_a.append(ads_b.append(ads_c))

##########################################################


X= all_ads.iloc[:,:-1].values
y = all_ads.iloc[:,-1].values

X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.25, random_state = 101)

sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
y_test= sc_y.fit_transform(y_test.reshape(-1,1))


pca=PCA(n_components = None)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

pca=KernelPCA(n_components = 7, kernel = 'rbf')
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

X_test_check = pd.DataFrame(X_train_pca).assign(y = y_train).corr()
######################################################
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train_pca, y_train.ravel())

y_pred_test = regressor.predict(X_test_pca)
y_pred_error = abs((y_pred_test  - y_test.ravel())/y_test.ravel())




