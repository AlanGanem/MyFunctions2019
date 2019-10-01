def density_plot(x,sticks = False,k1=1,k2=1,k3=1 ):
    from sklearn.neighbors import KernelDensity
    import pandas as pd
    from scipy.signal import argrelextrema
    from matplotlib import pyplot as plt    
    import numpy as np
    plt.clf()    
    
    iqr = float((pd.DataFrame(x).quantile(0.75))-(pd.DataFrame(x).quantile(0.25)))/1.349               
    m=min((k1*np.var(x))**(k2*1/2),k3*iqr/1.349)
    h=0.9*m/(len(x)**(1/5)) # h sugerido Stata
    x_d = np.linspace(min(x), max(x),len(x))
    kde = KernelDensity(bandwidth = h, kernel='gaussian')
    kde.fit(x[:, None])
    e = np.exp(kde.score_samples((x_d.reshape(-1,1))))
    mi, ma = argrelextrema(e, np.less_equal, order=1)[0], argrelextrema(e, np.greater_equal)[0]
    x_ = np.transpose([min(x)]*len(mi)+mi*(max(x)-min(x))/len(x_d))
    y_ = np.exp(kde.score_samples(([min(x)]*len(mi)+(mi)*((max(x)-min(x))/len(x_d))).reshape(-1,1)))
    sortId = np.argsort(x)
    plt.scatter(x_,y_, marker = 'x')
    x_split = np.split(np.sort(x),mi)
    logprob = kde.score_samples(x_d[:, None])
    plt.fill_between(x_d,e, alpha=0.5)
    plt.ylim((0, max(e))) 
    if sticks:    
        plt.plot(x, np.full_like(x, 0), '|k', markeredgewidth=1)
    plt.show()
    return
    