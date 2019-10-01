from sparse_dot_topn import awesome_cossim_topn as cosine_sim
import cython
from sparse_dot_topn import awesome_cossim_topn
import pickle
import os
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN, KMeans
from collections import Counter
g = os.path.join(os.path.dirname(r'C:\Users\ganem\Desktop\2Vintens\dados\Historical data\janeiro_2019\dados_tratados\\'), 'cv_matrix_janeiro')
cv_matrix = pickle.load(open(g, 'rb'))


g = os.path.join(os.path.dirname(r'C:\Users\ganem\Desktop\2Vintens\dados\Historical data\janeiro_2019\dados_tratados\\'), 'data_janeiro')
data_janeiro = pickle.load(open(g, 'rb'))




a= cv_matrix.astype('float64',copy=False)
a = normalize(a, norm='l2', axis=1,copy=False)

sim = cosine_sim(a,a.T, 1000, 0.7)
dist = sim.astype(bool,copy=False).astype('float64',copy=False)*0.99999-sim

data_janeiro_sim = data_janeiro[['ad_id','ad_title_corpus','category_id','price_min']].copy()
sims = [0.5,0.6,0.7,0.8,0.9]
for i in sims:
    sim1 = i
    min_size = 2
    clustering = DBSCAN(eps=1-sim1, min_samples=min_size, metric = 'cosine').fit(cv_matrix[:200000])
    data_janeiro_sim = eval('data_janeiro_sim.assign(%s = clustering.labels_)'% ('min_sim_'+str(int(i*100))))

group = data_janeiro_sim[data_janeiro_sim.min_sim_70== 11471]

Counter(data_janeiro_sim.min_sim_70).most_common()[0]
group = data_janeiro_sim[data_janeiro_sim.min_sim_70== 15]

