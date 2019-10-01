# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:35:06 2019

@author: ganem
"""

from collections import Counter, OrderedDict


cat = '421305_0'
clustered_data_janeiro_teste = clustered_data_janeiro_teste.assign(blueprint = '')
for cat in set(clustered_data_janeiro_teste.category_id):
    category = clustered_data_janeiro_teste[clustered_data_janeiro_teste.category_id.str.contains(cat)]
    words = []
    for i in range(len(category)):
        words += category.ad_title_corpus.iloc[i].split()
    c = len(words)
    dic =Counter(words)
    for key in dic:
        dic[key] = dic[key]/c
    dic = dic.most_common()
    
    nwords= 4
    cumulative = 0.5
    blueprint = []
    if cumulative:
        agregate = 0
        i = 0
        while agregate < cumulative:
            agregate+=dic[i][1]
            i+=1
            if agregate == 1:
                break
        i/len(set(words))
        for k in range(i):
            blueprint.append(dic[k][0])
    else:
        range_ = range(max(nwords, len(dic)))
        for k in range_ :
            blueprint.append(dic[k][0]) 
    cat_blueprint = ''
    for i in range(len(blueprint)):
        cat_blueprint += blueprint[i]+' '
    cat_blueprint = cat_blueprint[:-1]
    
    clustered_data_janeiro_teste[clustered_data_janeiro_teste.category_id.str.contains(cat)] = clustered_data_janeiro_teste[clustered_data_janeiro_teste.category_id.str.contains(cat)].assign(blueprint = cat_blueprint)



query = 'clustered_data_janeiro['
for palavra in blueprint:
    n= 'clustered_data_janeiro.ad_title_corpus.str.contains(%s%s%s, na= False)&' % ("'",palavra,"'")
    query+=n
query = query[0:-1]+']'


len(eval(query)[clustered_data_janeiro.category_id != cat])/len(category)
eval(query)[clustered_data_janeiro.category_id != cat][['category_id','ad_title_corpus']]

intersection = len(list(set(blueprint1).intersection(blueprint)))
union = (len(blueprint1) + len(blueprint)) - intersection
intersection/union








density_plot(torneira.price_min,sticks = True)




## NGRAMS
from collections import Counter
 from nltk import ngrams

ngram_counts = Counter(ngrams(bigtxt.split(), 2))
ngram_counts.most_common(10)



labels, values = zip(*Counter(words).most_common())
indexes = np.arange(len(labels))
width = 1

plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.show()


