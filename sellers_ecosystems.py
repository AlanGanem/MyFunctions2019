# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:52:33 2019

@author: PC10
"""
import sys
sys.path.append(r'C:\ProductClustering\my_functions')

import pandas as pd
from func_product_finder import pairwise_cosine_sparse_sim
import networkx as nx
from networkx.algorithms import bipartite
import community
import graphviz
from igraph import *
from matplotlib import cm
from func_get_data import get_data

import numpy as np
from matplotlib import  pyplot as plt
print('choose initial date (YYYY-MM-DD)')
init_date = input()
k1= 0.8


print("querying data")
data = get_data(free_query = '''
select
seller_id,
product_id,
min(price)*sum(daily_sales) / nullif(sum(daily_sales),0) as average_price,
greatest(max(sold_quantity)-min(sold_quantity),0) as sold_quantity,
sum(daily_sales*price) as seller_revenues, min(history.date) as initial_date
	from history
	join ad_static on history.ad_id = ad_static.ad_id
	join ads_scd on history.ad_id= ads_scd.ad_id and history.date >= ads_scd.initial_date and history.date < ads_scd.final_date
	where history.date > '{}' and history.price >= 120
	group by seller_id, product_id
	having sum(history.daily_sales*history.price)/nullif(count(history.daily_sales),0) > 50
         '''.format(init_date))
data = data[data['product_id']>0]
data['product_id'] = data['product_id'].astype(int)
print('generating products-sellers bipartite graph')
data = data.dropna()
data_dummies = pd.get_dummies(data.set_index('seller_id')[['product_id']].astype(int), columns = ['product_id'])
data_dummies = data_dummies.groupby(level= 'seller_id').sum()
arr = data_dummies.values
sim_matrix = pairwise_cosine_sparse_sim(arr,topn= 300,min_value = 0,normalize = False)


sources, targets = sim_matrix.nonzero()
tuples_list = list(zip(sources.tolist(), targets.tolist()))
tuples_list = [i for i in tuples_list if i[0] != i[1]]
g = Graph(tuples_list)
check = 0
while check != 'y':
    linear = True
    seller_id  = 97556253
    
    if linear:
        clusters= g.community_multilevel(weights = sim_matrix.data.astype(float))
    else:
        while True:
            print('select a value for k1 (sugestion: between [0.6 and 2])')
            try:
                k1 = float(input())
                break
            except:
                print('k1 must be a number (use point instead of comma as deciimal separator)')
        clusters= g.community_multilevel(weights = np.exp(k1*sim_matrix.data.astype(float)))
    membership = clusters.membership
    len(set(membership))
    ecossystems = pd.DataFrame(membership, columns = ['ecossystem'],index = data_dummies.index)
    ecossystem = ecossystems.loc[seller_id]['ecossystem']
    our_ecossystem = ecossystems[ecossystems['ecossystem'] == ecossystem]
    sellers_in_ecosystem = list(our_ecossystem.index.values)

    sellers = list(set(sellers_in_ecosystem))
    products = list(set(data[data['seller_id'].isin(sellers_in_ecosystem)]['product_id']))

    data_dummies_our_ecossystem_stacked_array = data[data['seller_id'].isin(sellers_in_ecosystem)][['seller_id','product_id','seller_revenues']].values
    range_len = range(len(data_dummies_our_ecossystem_stacked_array))
    list_of_weighted_edges_our_ecossystem = [tuple(data_dummies_our_ecossystem_stacked_array[i]) for i in range_len]

    O = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    O.add_nodes_from(sellers, bipartite=0)
    O.add_nodes_from(products, bipartite=1)
    # Add edges only between nodes of opposite node sets
    O.add_weighted_edges_from(list_of_weighted_edges_our_ecossystem)
    
    nodes = list(nx.bfs_tree(O,source = 97556253,depth_limit = 3).nodes())
    #donnot runthis line more than once
    O_root = O
    
    O = O_root.subgraph(nodes)
    sellers = {n for n, d in O.nodes(data=True) if d['bipartite']==0}
    products = set(O) - sellers
    
    sizes = []
    node_size ={}
    for node in O.nodes:
        if node == seller_id:
            sizes.append(300)
        else:
            sizes.append(O.degree([node],weight = 'weight')[node])
        node_size[node] = O.degree([node],weight = 'weight')[node]
    sizes=np.array(sizes)
    sizes = (sizes-min(sizes))/(max(sizes)-min(sizes))
    sizes = 400*sizes

    widths = []
    edge_size = {}
    for edge in O.edges:
        widths.append(O.get_edge_data(edge[0],edge[1])['weight'])
        edge_size[edge] = O.get_edge_data(edge[0],edge[1])['weight']
    widths=np.array(widths)
    widths = np.log1p(widths)
    widths = (widths-min(widths))/(max(widths)-min(widths))
    widths = 255*widths

    collors = ['g' if node == seller_id else 'r' if node in products else 'b' if node in sellers else 'UGA-BUGA' for node in O.nodes]
    assert 'UGA-BUGA' not in collors
    
    
    
    print('{} sellers'.format(len(sellers)))
    print('{} products'.format(len(products)))
    print('{} edges'.format(len(O.edges)))
    print('wish to print the graph (graphs with more than 1000 nodes tend to take longer to be plotted) ? [[y]/n]')
    draw_bool = input()
    if draw_bool == 'y':
        plt.clf()
        print('drawing graph')
        nx.draw_networkx(O,with_labels = True,node_size = sizes,node_color = collors,edge_color = widths, edge_cmap = cm.get_cmap('Greys'),font_size  = 12,font_color = 'g')
        print('list of product_ids:\n {}'.format(products))
        plt.show()
    print('whish to close?[[y]/n]')
    check = input()

i = 0
while i !='y':
    print('whish to close?[[y]/n]')
    i = input()


O = nx.bfs_tree(O,source = 97556253,depth_limit = 3)
len(tree.nodes())



                
    