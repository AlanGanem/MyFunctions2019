import pickle
from sklearn.preprocessing import normalize
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np
import os
import pandas as pd
from func_corpify import corpify
import re

def apply_word_embedings(titles,column_name = 'ad_title',model_folder_path = r'C:\ProductClustering\ProductsDB\fasttext_models\\' ,model_name = 'model_fast_text_sg_40'):
    
    titles = corpify(titles)
    g = os.path.join(os.path.dirname(model_folder_path), model_name)
    model_fast_text = pickle.load(open(g, 'rb'))
    vector_shape = model_fast_text.wv['teste'].reshape(1,-1).shape
    data = [] 
    for title in titles.ad_title_corpus:
        if len(title) != 0:
            try:
                data.append([normalize(np.sum(model_fast_text.wv[re.sub(r'[^a-zA-Z]',' ',title).lower().split()],axis = 0).reshape(1,-1))])
            except:
                embeds = []
                for word in title.split():
                    try:
                        embeds.append(normalize(np.sum(model_fast_text.wv[re.sub(r'[^a-zA-Z]',' ',word).lower().split()],axis = 0).reshape(1,-1)))
                    except:
                        embeds.append(np.zeros(vector_shape))
                data.append([np.average(np.array(embeds),axis = 0)])
        else:
            data.append([np.zeros(vector_shape)])
    
    titles = titles.assign(word_vector = data)
    titles = titles[titles.word_vector != '_EMPTY_']
    return titles

