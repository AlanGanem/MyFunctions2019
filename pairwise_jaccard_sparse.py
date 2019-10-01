from sparse_dot_topn import awesome_cossim_topn as dot_product
from scipy.sparse import csr_matrix
import numpy as np
from collections import Counter

def pairwise_jaccard_sparse(csr, topn=2000, min_value = 2):
    """Computes the Jaccard distance between the rows of `csr`,
    smaller than the cut-off distance `epsilon`.
    """
    csr= csr_matrix(csr).astype(bool,copy=False).astype(int,copy=False)
    csr = csr.astype('float64',copy=False)
    csr_rownnz = csr.getnnz(axis=1)
    intrsct = dot_product(csr,csr.T, topn, min_value-0.1)

    nnz_i = np.repeat(csr_rownnz, intrsct.getnnz(axis=1))
    unions = nnz_i + csr_rownnz[intrsct.indices] - intrsct.data
    intrsct.data = 1.0 - intrsct.data / unions
       
    return intrsct



g = os.path.join(os.path.dirname(r'C:\ProductClustering\output_data\\'), 'cv_matrix')
cv_matrix = pickle.load(open(g, 'rb'))

clustering = DBSCAN(eps=1-0.68, min_samples=3, metric = 'precomputed').fit(intrsct)

Counter(clustering.labels_).most_common()[0]
data_janeiro = data_janeiro.assign(cluster_id = clustering.labels_ )