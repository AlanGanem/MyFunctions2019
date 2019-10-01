import pandas as pd
import pickle
import os



cat_tree = pd.read_csv(r'C:\ProductClustering\input_Data\categorytree.csv')

for column in list(cat_tree):
    cat_tree = eval('cat_tree.assign(%s = cat_tree.%s.str.replace("MLB",""))' %(column,column))

lvl5={}
for father_category in set(cat_tree.category_0):
    lvl5[str(father_category)] = cat_tree[cat_tree.category_0 == father_category][~cat_tree.category_5.isnull()].iloc[:,5].tolist()
lvl4={}
for father_category in set(cat_tree.category_0):
    lvl4[str(father_category)] = cat_tree[cat_tree.category_0 == father_category][~cat_tree.category_4.isnull()].iloc[:,4].tolist()
lvl3={}
for father_category in set(cat_tree.category_0):
    lvl3[str(father_category)] = cat_tree[cat_tree.category_0 == father_category][~cat_tree.category_3.isnull()].iloc[:,3].tolist()
lvl2={}
for father_category in set(cat_tree.category_0):
    lvl2[str(father_category)] = cat_tree[cat_tree.category_0 == father_category][~cat_tree.category_2.isnull()].iloc[:,2].tolist()
lvl1={}
for father_category in set(cat_tree.category_0):
    lvl1[str(father_category)] = cat_tree[cat_tree.category_0 == father_category][~cat_tree.category_1.isnull()].iloc[:,1].tolist()

all_lvls = {}
for key in list(lvl5):
    lvl5[key].extend(lvl4[key])
    lvl5[key].extend(lvl3[key])
    lvl5[key].extend(lvl2[key])
    lvl5[key].extend(lvl1[key])    
    all_lvls[key] = lvl5[key]
    
for key in all_lvls:
    all_lvls[key] = list(set(all_lvls[key]))



counter = 0
for key in list(all_lvls.keys()):
    print(len(data_janeiro[data_janeiro.category_id.isin(all_lvls[key])]))
    if len(data_janeiro[data_janeiro.category_id.isin(all_lvls[key])]) == 0:
        print(len(data_janeiro[data_janeiro.category_id == int(key)]))
        counter+=len(data_janeiro[data_janeiro.category_id == int(key)])
    counter+=len(data_janeiro[data_janeiro.category_id.isin(all_lvls[key])])




################ unwanted categories

category_id = '1430'

unwanted_cats = '(' + ','.join(all_lvls[category_id]) + ')'

g = open(os.path.join(os.path.dirname(r'C:\ProductClustering\sql_queries\categories\\'), 'cat_dic'), 'wb')
pickle.dump(all_lvls,g)

g.close()

text_file = open("vestuário.txt", "w")
text_file.write("unwanted_cats")
text_file.close()

text_file = open("vestuário.txt", "r")

