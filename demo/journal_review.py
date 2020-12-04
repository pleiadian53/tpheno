import plotly.plotly as py
import plotly.graph_objs as go

# import matplotlib.pyplot as plt 

# non-interactive mode
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt


import pandas as pd
from pandas import Series, DataFrame, TimeSeries
import numpy as np

def load(**kargs):

    basedir = kargs.get('inputdir', os.getcwd())
    fname = kargs.get('fname', 'fulldataset.csv')
    fpath = os.path.join(basedir, fname)
    assert os.path.exists(fpath), "Invalid input path: %s" % fpath
    df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)

    print('load> dataframe dim: %s' % str(df.shape))

    # t_analyis(df)

    return df

def t_plot_df(df=None, **kargs):

    plt.clf()
    if df is None: 
    	df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])

    outputdir = os.path.join(os.getcwd(), 'plot')
        
    # fig = df.plot(x='docID', y='hit_ratio', kind='hist') 
    fig = df.plot(x='a', y='b', kind='bar') 

    if not os.path.exists(outputdir): 
        print('save_plot> creating new directory: %s' % outputdir)
        os.mkdir(outputdir)
    fpath = os.path.join(outputdir, fname)
    plt.savefig(fpath) 
    print('save_plot> saved label test result to %s' % fpath)

    return 

def t_analyis(df, **kargs): 
    plt.clf()

    # scatter plot between Trial and Comment? 
    trialIDs = df['Trial'].values
    commentIDs = df['Comment'].values 

    overlapIDs = set(trialIDs).intersection(commentIDs)
    print('> any overlaps? total=%d, overlap=%d' % (len(trialIDs), len(overlapIDs)))


    plt.figure()
    # scatter plot between TrialDate and CommentDate
    df.plot.scatter(x='TrialDate', y='CommentDate')

    plt.show()
    
    return


def test(**kargs): 
    basedir = '/Users/pleiades/Documents/work/time_series/data'
    fname = 'fulldataset.csv'
    df = load(inputdir=basedir, fname=fname)
    t_analyis(df, **kargs)

    return

if __name__ == "__main__":
    test()  