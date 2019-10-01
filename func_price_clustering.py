import time
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
import pandas as pd
from scipy.stats import variation as var_coef
from KDEpy.NaiveKDE import NaiveKDE
import pandas as pd
import time

###################################################################
def price_clustering_apply(segmentation_policy, dataframe,column='category_id',column_name = 'price_min'):
    
    s = time.time()
    data=dataframe.copy()   
    unique_categories=list(set(segmentation_policy))
    for i in unique_categories:
        unchanged_cat_str = str(i)
        range_ = range(len(segmentation_policy[unchanged_cat_str]))
        for j in range_:
            if j == min(range_):
                indexes = data[data.category_id == int(unchanged_cat_str)].index[data[data.category_id == int(unchanged_cat_str)][column_name].between(0,max(segmentation_policy[unchanged_cat_str][j]),inclusive=True)]
            elif j == max(range_):
                indexes = data[data.category_id == int(unchanged_cat_str)].index[data[data.category_id == int(unchanged_cat_str)][column_name].gt(min(segmentation_policy[unchanged_cat_str][j-1]))]
            else:
                indexes = data[data.category_id == int(unchanged_cat_str)].index[data[data.category_id == int(unchanged_cat_str)][column_name].between(min(segmentation_policy[unchanged_cat_str][j]),max(segmentation_policy[unchanged_cat_str][j]))]
            new_category = str(unchanged_cat_str + '_' + str(j))
            data.loc[indexes,'category_id'] = new_category
    data.category_id = data.category_id.astype(str)
    print('ran in '+ str(time.time()-s)+'s')
    return data


################################################################
def price_clustering(dataframe, min_items = 2, min_var_coef = 0.3,column='product_id', fluctuation = 0.20, column_name = 'price_min'):
    
    #categories_list = [categories_list[i] for i in range(len(categories_list)-1) if len(categories_list[i]) >= min_items]
    data = dataframe.copy()
    s = time.time()
    new_cat_counter = 0
    unique_categories= list(set(data[column]))
    segmentation_policy={}
    for i in unique_categories:
        x = np.sort(data[data[column] == i][column_name].values)
        var_coef_x = var_coef(x)
        iqr = float((pd.DataFrame(x).quantile(0.75))-(pd.DataFrame(x).quantile(0.25)))/1.349
        # instantiate and fit the KDE model
        varx = np.var(x)
        k1=1
        k2=1
        k3=1
        m = min(k1*varx**(k2*1/2) , k3*iqr/1.349)
        h = 0.9*m/(len(x)**(1/5)) # h sugerido Stata
        if (h > 0)&(len(x)>2):
            if var_coef_x >= min_var_coef:
                x_d = np.linspace(min(x), max(x),len(x))
                kde = KernelDensity(bandwidth = h, kernel='gaussian')
                kde.fit(x[:, None])
                e = np.exp(kde.score_samples((x_d.reshape(-1,1))))
                mi = argrelextrema(e, np.less)[0]
                #ma = argrelextrema(e, np.greater)[0]
                x_split = np.split(np.sort(x),mi)
                #k = 1/len(x_split)*len(x)/(np.var(x))**(1/2)
                x_band = []
                for k in x_split:
                    x_band.extend(np.repeat(np.median(k)*fluctuation,len(k)))
                x_band = np.array(x_band)
                #estimator = NaiveKDE(kernel='gaussian', bw=fluctuation/np.log(k*x)*x).fit(np.array(x))
                estimator = NaiveKDE(kernel='gaussian', bw=x_band).fit(np.array(x))
                x_d = np.linspace(min(x), max(x),len(x))
                y = estimator.evaluate(x_d)
                mi = argrelextrema(y, np.less)[0]
                x_split = np.split(np.sort(x),mi)
                segmentation_policy[str(i)] = x_split
                j=0

                unchanged_cat_str = str(int(i))
                while j in range(len(x_split)):
                    indexes = eval('data[data.%s == i].index[data[data.%s == i][column_name].isin(x_split[j])]' % (column,column))
                    new_category = str(unchanged_cat_str + '_' + str(j))
                    data.loc[indexes,column+'_by_price'] = new_category
                    j+=1
                    new_cat_counter+=1
            else:
                unchanged_cat_str = str(int(i))
                new_category = str(unchanged_cat_str + '_0')
                indexes = eval('data[data.%s == i].index' % column)
                data.loc[indexes,column+'_by_price'] = new_category
                new_cat_counter+=1

        else:
            if var_coef_x <= min_var_coef:
                unchanged_cat_str = str(int(i))
                new_category = str(unchanged_cat_str + '_0')
                indexes = eval('data[data.%s == i].index' % (column))
                data.loc[indexes,column+'_by_price'] = new_category
                new_cat_counter+=1
            else:
                unchanged_cat_str = str(int(i))
                index_max = eval('data[data.%s == i].index[data[data.%s == i][column_name] == max(data[data.%s == i][column_name])]' % (column,column,column))
                index_min = eval('data[data.%s == i].index[data[data.%s == i][column_name] == min(data[data.%s == i][column_name])]' % (column,column,column))
                new_category_min = str(unchanged_cat_str + '_0')
                data.loc[index_min,column+'_by_price'] = new_category_min
                new_category_max = str(unchanged_cat_str + '_1')
                data.loc[index_max,column+'_by_price'] = new_category_max

    print('ran in '+ str(time.time()-s)+'s')
    print(str(new_cat_counter) +' new categories found')
    newdata= data
    return [newdata, segmentation_policy]
