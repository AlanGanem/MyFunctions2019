import fastcluster
from collections import  Counter
from sklearn.cluster  import MiniBatchKMeans
import numpy as np
import fastcluster
import scipy
from func_search_engine import search_engine

a= ranking_fevereiro[ranking_fevereiro.title_corpus.str.contains('suco laranja')]
a = clustered_data_fevereiro[clustered_data_fevereiro.product_id == 36986]


cluster_ = fastcluster.linkage_vector(data)
cluster_labels = scipy.cluster.hierarchy.fcluster(cluster,0.05, criterion  = 'distance')
ranking_fevereiro= ranking_fevereiro.assign(a =cluster_labels)

from func_cosine_sparse  import cosine_sparse
from scipy.sparse import csr_matrix

data = np.array([i[0][0]for i in ranking_fevereiro.product_word_vector])
data = cosine_sparse(csr_matrix(data),min_value = 0.85)
data = data.A




ranking_fevereiro= ranking_fevereiro.assign(a =cluster_labels)
ranking_fevereiro[ranking_fevereiro.a == np.argmax(ranking_fevereiro.groupby('a').count().product_id)].product_name
