
# Note
# ----
# 
#
# Todo List 
# ---------
# 1. Add project operations and options (to load_df(), etc.) ... 3.5.2015
#
#

from pandas import DataFrame, Series
import pandas as pd 
import numpy as np
import sys, os, re, math

import predicate

# try: 
#     from pudb import set_trace; set_trace()
# except: 
#     print "dfUtils::test> pubd is not installed."

# user-defined
from utils import *
import utils

def getAvgEffAttributes(df, trivial_val=0):
    """
    Analyze sparse dataframe and determine average number of 
    non-trival/active attributes, etc. 
    """
    s = 0
    dfc = df.apply(pd.Series.value_counts, axis=1)
    try: 
        # s = df.apply(pd.Series.value_counts, axis=1)[trivial_val].sum()  # no need to invoke .dropna() prior to sum()
        s = dfc[trivial_val].sum()
    except: 
        if not trivial_val in dfc.columns: 
            print('warning> trival value %s is not in the data at all!' % trivial_val)
        else: 
            raise ValueError, "unknown error with trivail value %s" % trivial_val

    # total active values
    neff = df.size - s
    nrow = df.shape[0] 

    return neff/(nrow+0.0)

def add_metadata(df, **kargs):
    """

    [metadata]
    fields: 
      targets: e.g. the ICD9 codes you are looking after 
      table: e.g. diag  name of the table 
      n_patients: number of (unique) patients in the dataframe 
      n_records: number of records

    """
    df.meta = {}
    for k, v in kargs.items(): 
        df.meta[k] = v
    return df
def add_metadata2(df, params): 
    assert isinstance(params, dict)
    df.meta = {}
    for k, v in params.items(): 
        df.meta[k] = v
    return df

def display_metadata(df): 
    try: 
        df.meta 
    except: 
        print "display> no meta associated with the input dataframe"
    output = ", ".join("{0}={1}".format(k, v)
                      for k, v in df.meta.items())
    return output

def display(df, verbose=True, n=None):
    if df is None or df.empty: 
        if verbose: print('display> Empty dataframe.')
        return
    df2 = df if n is None else df[:n] 
    try: 
        print df2.to_string(index=False) 
    except Exception, e: 
        try: 
            if verbose: print('Warning: Had trouble converting to string: %s' % e)
            print df2
        except: 
            print('Warning: Could not display the input dataframe') 
    return

def drop_nan(df, col='mrn'): 
    """

    Memo
    ----
    df.replace([np.inf, -np.inf], np.nan).dropna(subset=["col1", "col2"], how="all")
    """
    return df[pd.notnull(df[col])]

def convert_dtype(df, col='mrn', typ='int64', _debug=0, drop_nan_=False, no_op=True): 
    """
    Convert (in place) the data type for specified column of the dataframe. 

    [debug]
    1. A value is trying to be set on a copy of a slice from a DataFrame.
       Try using .loc[row_indexer,col_indexer] = value instead
       See the the caveats in the documentation: 
       http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
       > df[col] = df[col].astype(typ)
    """
    # [todo] astype of whatever df's mrn is
    msg = ''
    try: 
        # if drop_nan_: df = df[pd.notnull(df[col])]
        df[col] = df[col].astype(typ)
        return df
    except KeyError, e:  
        msg = "convert_dtype> column %s does not exist" % col
        if not no_op: 
            raise KeyError, msg
    except Exception, e: 
        try:     
            try: 
                n0 = df.shape[0]
                msg = "convert_dtype> 'df[%s]' may contain NaN values. Try removing rows with NaNs\n" % col
                df = df[pd.notnull(df[col])]
                msg += "convert_dtype> size of df %d -> %d\n" % (n0, df.shape[0])
                df[col] = df[col].astype(typ)
                if _debug: div(message=msg, symbol='*', adaptive=False)
                return df 
            except: 
                msg = "convert_dtype> Could not cast 'df[%s]' to type: %s\n" % (col, typ)
                # if _debug > 0: 
                msg += "  + %s\n" % e 
                msg += "  + value:\n%s" % df.head(3)

        except Exception, e: 
            msg = "convert_dtype> %s\n" % e
            msg += "  + Invalid dataframe:\n%s" % str(df.head())
    if _debug: print msg
    if no_op: 
        # do nothing
        return df
    raise RuntimeError, msg 
   
def df_size(df):
    """Return the size of a DataFrame in Megabyes"""
    total = 0.0
    for col in df:
        total += df[col].nbytes
    return total/1048576.0


def convert_dtypes(df, dtypes, drop_nan_=False):
    assert isinstance(dtypes, dict), "convert_dtypes> invalid dtypes: %s" % dtypes
    for col, typ in dtypes.items(): 
        df = convert_dtype(df, col, typ, drop_nan_=drop_nan_)
    return df

def match0(df, criteria): 
    """
    Find rows that match given criteria. 

    Memo
    ----
    1. This doesn't work!!!! because????
    """
    assert isinstance(criteria, dict) 

    df = DataFrame()   
    for k, v in criteria.items(): 
        if not (k in df.columns): break
        try: 
            df = df.loc[df[k]==v]
        except: 
            if predicate.isNull(k) or predicate.isNull(v): 
                print("match> Warning: null criteria: %s: %s. Assuming no match." % (k, v))
                df = DataFrame()
            else: 
                raise RuntimeError, "match> unknown error. criteria: %s, df:\n%s\n" % (criteria, df.head())

        if df.empty: break 
    return df

def match(df, criteria): 
    """
    Find rows that match given criteria. 
    """
    assert isinstance(criteria, dict)   
    # [note]
    # df[ df['mrn']==x & df['label'] == y ] 
    for k, v in criteria.items(): 
        # if not (k in df.columns): continue
        if not (k in df.columns): raise ValueError, "match> %s is not present in %s" % (k, df.columns.values)
        df = df.loc[df[k]==v]    # df.loc[idx] selects rows with indices as those in idx
        # df = df[df[k]==v]    # this also works! 
        if df.empty: break 
    return df

def spec(df, prompt=None): 
    """

    Related 
    ------- 
    profile()
    """
    if prompt is None: prompt = whosdaddy() 
    print "%s> header: %s" % (prompt, df.columns)
    print "%s> dim: %s" % (prompt, str(df.shape)) 
    print "%s> dtypes:\n%s" % (prompt, df.dtypes)

    return 

def isNullData(df):
    return df is None or df.empty 
isNullSet = isNullData  # [alias]

def support(df, target_field='target'): 
    labels = df[target_field].values
    s = {}
    for label in labels: 
        if not s.has_key(label): s[label] = 0 
        s[label]+=1 
    return s

def hasAllSupports(df, labels, min_support=0):  # has all labeled data
    if isNullData(df): return False
    s = support(df)
    # assert len(labels) > 0 
    for label in labels:
        n = s.get(label, 0) 
        if n <= min_support: 
            return False 
    return True
hasSupport = hasAllSupports  # [alias]

def drop(ts, fields): 
    assert hasattr(fields, '__iter__')
    for field in fields: 
        if field in ts.columns: 
            ts = ts.drop(field, axis=1)
    return ts 

def profile(df, target_field='target', index_field='mrn'): 
    """
    
    Attributes
    ----------
    nrow 
    nrow/size, ncol
    dim 
    labels: all unique labels
    support: type {}; number of instances for each label

    Memo
    ----
    nrow, ncol, size, dim, n_patients, n_unique, labels, support
    """
    assert isinstance(df, DataFrame) # ok even if df is not an 'instance variable' per se
    fparams = {}
    fparams['nrow'] = fparams['size'] = df.shape[0]
    fparams['ncol'] = df.shape[1]
    fparams['dim'] = df.shape

    try: 
        fparams[Params.Stats.n_patients] = fparams['n_unique'] = len(set(df[Params.index_field].values))
    except: 
        print('> index field %s in dataframe? %s' % (Params.index_field, Params.index_field in df.columns))

    labels = None
    print "profile> dim of input df: %s" % str(df.shape)
    assert target_field in df.columns
    try: 
        labels = df[target_field].values # np.unique(df[target_field])
    except: 
        print "dfUtils.profile> could not extract labels."
    if labels is not None:
        ulabels = np.unique(labels)
        fparams['labels'] = ulabels
        fparams['support'] = dict.fromkeys([str(l) for l in ulabels], 0) 
        for label in ulabels: 
            fparams['support'][str(label)] = fparams[str(label)] = list(labels).count(label)
    return fparams  

### dataframe manipulations 

def select_by_index(root, index=0, protocol=None, identifier='default', ext='csv'): 
    """
    Select all the files (e.g. training set files) sharing the same index (which are collectively 
    used to train a classifier). So if there are N indices, we will end up having 
    N groups of files, each of which are used to train a separate classifier. 

    Helper function for combine(). 

    Arguments
    ---------
    protocol: a tuple forming the regex for files with the first element representing a template 
              (of string type) for file name and the reset of the element representing arguments to 
              the template. 

        e.g.  bt-100-481-2.csv 


    """
    def file_regex(): 
        if protocol is not None: 
            assert hasattr(protocol, '__iter__') 
            t = protocol[0]
            if len(protocol) > 1: 
                t = t % protocol[1:]
            return re.compile(r'%s' % t)
    def is_matched(f, p=None):
        if p is not None: 
            if p.match(f): 
                return True
            else: 
                # print('test> file %s (type %s) does not match %s' % (f, type(f), p.pattern))
                return False 
        if identifier is None: 
            return True
        return f.find(identifier) >= 0 and f.find(str(index)) >= 0
    def gen_files(): 
        for f in glob.glob(os.path.join(root, '*.%s' % ext)):
            if is_matched(os.path.basename(f)):
                yield f

    import glob, re
    if ext.startswith('.'): ext=ext[1:]
    
    # p = file_regex() # protocol is not a list any more ... 11.15.15
    p = re.compile(protocol)
    print('dfUtils-test> file pattern: %s' % p.pattern)
    candidates = []
    for f in glob.glob(os.path.join(root, '*.%s' % ext)):
        fn = os.path.basename(f)
        # print('> %s' % fn)
        if is_matched(fn, p):
            candidates.append(f)
    return candidates

def combine(root, files=None, fname=None, identifier=None, criteria=None, ext='csv', 
              n_samples=None, shuffle=True, dtypes=None, sep=',', _save=True, **kargs):
    """
    Combine a set of csv files with the same header.

    Arguments
    ---------
    fname: the name given to the combined csv file
    identifier: a string used as a matching keyword for 
                the candidate files
    files: if given, only combine these files

    **kargs: 
      fn_sep: file name separator 

    """ 
    return

def split(df, test_size=0.3, shuffle=False): 
    if shuffle: df = df.reindex(np.random.permutation(df.index))
        
    # number of examples
    n_total = df.shape[0]
    if test_size is not None: 
        if test_size > 1.0: test_size = 1.0 
            # if test_size < 0.005: test_size = 0.005   # [todo]
        n_samples = math.floor(n_total * (1.0-test_size))
        if n_samples > 0: 
            return (df[:n_samples], df[n_samples:])

    return (df, DataFrame(columns=df.columns))

def subset(df, n_samples=10, ratio=None, shuffle=True):
    if shuffle: df = df.reindex(np.random.permutation(df.index))
    if n_samples is not None: 
        return df[:n_samples]
    if ratio is not None: 
        return split(df, test_size=(1.0-ratio))[0]
    return df

def subset2(path, n_samples=None, ratio=0.8, min_ratio=0.05, sep='|', shuffle=True, verbose=False, save_=True): 
    """
    Take a subset (of entries) of a csv file.  
    """
    root, fname = os.path.dirname(path), os.path.basename(path)
    df = load_df(_file=path, from_csv=True, sep=sep)
    if verbose: print('subset2> loaded dataframe of dim %s from %s' % (str(df.shape), path))
    
    n_total = df.shape[0]
    if n_samples is None: 
        n_samples = math.floor(n_total*ratio)
    min_samples = math.floor(n_total*min_ratio) if min_ratio is not None else 0
    if n_samples < min_samples: 
        if verbose: print('subset2> n_samples:%d may be too small; adjusting to %d' % (n_samples, min_samples))
        n_samples = min_samples
    if shuffle: df = df.reindex(np.random.permutation(df.index))
    df = df[:n_samples]
    
    if save_:
        # rename file 
        fname_prefix, ext = os.path.splitext(fname)
        fname = (fname_prefix + '-%s' % n_samples) + ext
        path = os.path.join(root, fname) 
        if verbose: print('subset2> saving new dataframe of dim %s to %s' % (str(df.shape), path))
        save_df(df, _file=path, is_data=True, to_csv=True) 
    return df

def subset3(path, n_samples=None, balanced=True, ratio=0.8, min_ratio=0.05, sep=',', shuffle=True, 
             verbose=False, save_=True, verify_=False): 
    """
    Take a (balanced) subset of a training set data. 
    """
    from utils import div
    import sys
    from pprint import pprint
    if sep != Params.sep_tset: div(message="Warning: a training set data are usually '%s'-separated." % Params.sep_tset, symbol='~')

    root, fname = os.path.dirname(path), os.path.basename(path)
    df = load_df(_file=path, from_csv=True, sep=sep)
    params = profile(df)
    nrow, ncol = params['nrow'], params['ncol']
    assert (nrow, ncol) == df.shape and nrow >= 2 
    
    # determine n_samples 
    n_total = nrow 
    if n_samples is None: n_samples = math.floor(n_total*ratio)
    min_samples = math.floor(n_total*min_ratio) if min_ratio is not None else 0
    if n_samples < min_samples: 
        if verbose: print('subset3> n_samples:%d may be too small; adjusting to %d' % (n_samples, min_samples))
        n_samples = min_samples

    # how many instances should a label have? 
    labels = params['labels']
    dfl = []
    dimtb = {}
    min_nrow, max_nrow = (np.inf, -np.inf)
    n_avg = int(n_samples/(len(labels)+0.0)) # each label should have this many instances if obtainable 
    use_min_nrow = False
    for label in labels: 
        n = params[str(label)] 
        dimtb[str(label)] = n  # each label has n instances 
        if balanced and n < n_avg: # if any of the num. of instances does not meet the average, then 'optimal' balanced set is not possible
            use_min_nrow = True
            if verbose: 
                msg = 'Warning: %s-labeled data are not sufficient to reach a balanced data set given n_samples=%d' % (label, n_samples)
                div(message=msg, symbol='%')
        if n <= min_nrow: min_nrow = n
        if n >= max_nrow: max_nrow = n
    if verbose: 
        msg = "subset3> n_avg: %d, min_nrow: %d, max_nrow: %d" % (n_avg, min_nrow, max_nrow)
        div(message=msg, symbol='~')
    for label in labels:
        try: 
            subdf = df[df[Params.target_field]==label]   
        except: 
            print('subset3> label %s is not in %s?' % (label, [c for c in df.columns]))
            sys.exit(1)
        if verbose: print('subset3> prior to slicing, subset of df with label=%s has dim: %s' % (label, str(subdf.shape)))
        if shuffle: subdf = subdf.reindex(np.random.permutation(subdf.index))
        if use_min_nrow: 
            subdf = subdf[:min_nrow]
        else: 
            subdf = subdf[:n_avg]
        dfl.append(subdf)

    df = pd.concat(dfl, ignore_index=True)
    if shuffle: df = df.reindex(np.random.permutation(df.index))
    if verbose: print('subset3> slicing and combining completed, dim of new tset: %s' % str(df.shape))
    if verify_: 
        params = profile(df)
        print("subset3> profile of new training set:")
        div(); pprint(params); div()
        if balanced: 
            nref = params[str(labels[0])]
            for label in labels[1:]: 
                assert nref == params[str(label)], "imbalanced training data. see profile above." 

    if save_:
        # rename file 
        fname_prefix, ext = os.path.splitext(fname)
        fname = (fname_prefix + '-%s' % n_samples) + ext
        path = os.path.join(root, fname) 
        if verbose: print('subset3> saving new tset of dim %s to %s' % (str(df.shape), path))
        save_df(df, _file=path, is_data=True, to_csv=True, sep=Params.sep_tset) 

    return df

### sampling 

def sample(df, n):
    import random
    # df.ix[np.random.choice(df.index, n)]
    # df.sample(n=n)
    return df.ix[random.sample(df.index, n)]

def sample_class(df, n): 
    return df

### predicates
def is_aligned(ts1, ts2, field='target', verbose=True):
    if isinstance(field, dict):
        assert all([f in ts1.columns and f in ts2.columns for f in field.keys()])
        for f, typ in field.items():
            if verbose: print("is_aligned> verifying field %s via type: %s" % (f, typ)) 
            ts1 = convert_dtypes(ts1, typ)
            ts2 = convert_dtypes(ts2, typ)
            if not all(ts1[f] == ts2[f]): 
                return False 
        return True
    elif hasattr(field, '__iter__'): 
        if verbose: print "is_aligned> verifying fields: %s" % field
        assert all([f in ts1.columns and f in ts2.columns for f in field])
        for f in field: 
            if not all(ts1[f] == ts2[f]): 
                return False
        return True
    else: 
        assert field in ts1.columns and field in ts2.columns
    return all(ts1[field] == ts2[field]) 

def to_dict(df, x, y):  # feature/medcode to description
    assert x in df.columns and y in df.columns 
    return dict(zip(df[x], df[y]))    


def test_manipulate(**kargs): 
    """

    Log
    --- 
    * ./data-exp/cdr/lab/cerner/cerner_blood_481_tset_mixed.csv
    * 'data-lab/cerner/cerner_blood_481_tset_mixed.csv'
    * 'data-meds/cerner_antibiotic_481_tset_mixed.csv'

    Data
    ----
    * ./data-exp/cdr/lab/cerner/cerner_microbio_tset_mixed_infections_bt.csv

    """
    import os 
    from utils import div
    from learner import Group, Feature
    from pprint import pprint
    
    file_ = kargs.get('file_', os.path.join(ProjDir, 'data-exp/cdr/lab/cerner/cerner_microbio_tset_mixed_infections_bt.csv'))
    print('test> path: %s' % file_)
    df = load_df(_file=file_, from_csv=True, sep=',')

    # profiling
    params = profile(df)
    div(); pprint(params); div()

    # df.columns is an index object
    df = Group.canonicalize(df)  # [w1][1]
    fg = Feature(df.columns)
    # print "> total feature set: %s" % fg.total()
    print("> number of features: %d =?= %d, type: %s" % (len(fg.total()), len(fg.active()), type(fg.total()) ))
    print("> number of columns:%s type: %s, examples: %s" % (len(df.columns), type(df.columns), df.columns))
    div()

    # check support and indexing 
    columns = Series( [f for f in df.columns[:10]] )
    print("> ncols: %d, type: %s, ex: %s" %(len(columns), type(columns), columns))
    idx = [1, 3, 5]
    support = [False] * len(columns); 
    for i in idx: 
        support[i] = True
    print("> idx: %s -> features:\n %s" % (idx, columns[idx]))
    print("> support: %s -> features:\n %s" % (support, columns[support]))

    return

def test_subsetting():
    path = './data-diag/diag_112.9.csv' 
    # path = './data-diag/diag_no-infections-as-0.csv'
    subset2(path=path, n_samples=5000, verbose=True)

    # subsetting training set?
    # path = './data-exp/cdr/lab/cerner/cerner_microbio_tset_mixed_infections_bt.csv'
    # subset3(path=path, n_samples=5000, verbose=True, verify_=True)

    return 

def t_combine(): 
    """
    Boilerplate/test code for combine-related operations. 
    """
    import os
    root = os.path.join(os.getcwd(), 'train')
    identifier = 'bt-100'
    n_iter = 5 

    # this regex doesn't work if .pattern is passed as an argument
    # p_icd9 = re.compile(r'(?P<char>v|e)?(?P<base>\d+(?![-]))(\.(?P<decimal>\d+))?', re.I) 
    
    p_icd9 = re.compile(r'(E|V)?\d+(\.\d+)?', re.I)
    pat = '%s-%s-%s'
    for i in range(n_iter): 
        files = select_by_index(root, index=i, protocol=(pat, identifier, p_icd9.pattern, i), identifier=identifier, ext='csv')
        print("Found %d files with index %d:\n%s" % (len(files), i, files))

def t_profile(path, files, **kargs):
    """
    Boilerplate/test code for profiling mulltiple files. 
    """ 
    sep = kargs.get('sep', ',')
    sizes = []
    ncols = []
    for f in files: 
        fp = os.path.join(path, f)
        assert os.path.exists(fp)  
        df = load_df(_file=fp, from_csv=True, sep=sep, verbose=False) 
        fparams = profile(df)
        sizes.append(fparams['nrow'])
        ncols.append(fparams['ncol'])
    print 'sizes: %s' % sizes 
    print 'ncols: %s' % ncols

    return

def t_typecast(sep=','):
    import predicate
    index_field = 'mrn'
    f = '/phi/proj/poc7002/bulk_training/data-exp/cdr/lab/cerner/cerner_microbio_112.0_tset_bt_100.csv'
    df0 = load_df(_file=f, from_csv=True, sep=sep, verbose=False) 
    idx = set(df0[index_field].values)
    if all([predicate.isNumber2(e) for e in idx]): 
        print('> normal.')
    else: 
        print('> non-numerical values found.')
    df = load_df(_file=f, from_csv=True, sep=sep, verbose=False) 

    # cannot convert type because some entries are nan
    # df = df.dropna(thresh=1)
    # df = df[pd.notnull(df['mrn'])]
    df =convert_dtype(df, col='mrn', typ='int64', _debug=1)

    s0 = set(df[index_field].values)
    s = set([e for e in df[index_field].values if predicate.isNumber(e)])
    print('> filter nan rows after convert_dtype: %d >=? %d' % (len(s0), len(s)))
    idx2 = set(s)
    div(message='> size of idx: %d' % len(idx2))
    if all([isinstance(e, int) for e in idx2]): 
        print('> normal.')
        print('> examples: %s' % list(idx2)[:5]) 
    else: 
        print('> non int found.')  
        print('> examples: %s' % list(idx2)[:5])  


    return

def t_filter(): 
    from numpy.random import randn
    div('(1) demo dropping rows with nan ...', border=1)
    df = DataFrame(randn(10,3),columns=list('ABC'),index=pd.date_range('20130101',periods=10))
    df.ix[6,'A'] = np.nan
    df.ix[6,'B'] = np.nan
    df.ix[2,'A'] = np.nan
    df.ix[4,'B'] = np.nan
    print("> show row 0-5:\n%s\n" % df.iloc[0:6])
    df2 = df.iloc[0:6].dropna()
    print("> after dropping rows with nan:\n%s\n" % df2)

    div('(2) filtering out data without NaN ...', border=1)
    df = pd.DataFrame({'movie': ['thg', 'thg', 'mol', 'mol', 'lob', 'lob'],
                  'rating': [3., 4., 5., np.nan, np.nan, np.nan],
                  'name': ['John', np.nan, 'N/A', 'Graham', np.nan, np.nan]})
    print("> df:\n%s\n" % df)
    nbs = df['name'].str.extract('^(N/A|NA|na|n/a)')  # standardize nan data
    nms=df[(df['name'] != nbs) ]
    print('> nms:\n%s\n' % nms)
    thresh = 2 
    nms = nms.dropna(thresh=thresh)
    print('> after dropping rows with at least 2 %d nan:\n%s\n' % (thresh, nms))
    div()
    nms01 = nms[nms.name.notnull()]
    print('> dropped rows with na in name col:\n%s\n' % nms01)
    # nms02 = nms[np.isfinite(nms['name'])] 
    #  => error: ufunc 'isfinite' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
    # print('> dropped rows with na in name col:\n%s\n' % nms02)
    nms03 = nms[pd.notnull(nms['name'])]
    print('> dropped rows with na in name col:\n%s\n' % nms03)

    return 

def t_dtype(sep=','): 
    import predicate
    index_field = 'mrn'
    f = '/phi/proj/poc7002/bulk_training/data-exp/cdr/lab/cerner/cerner_urine_009.2_tset_bt_77.csv'
    df0 = load_df(_file=f, from_csv=True, sep=sep, verbose=False) 
    idx = set(df0[index_field].values)
    print("> Should expect to see integer types for idx: %s" % list(idx)[:5])

    a = [['a', '1.2', '4.2'], ['b', '70', '0.03'], ['x', '5', '0']]
    df = pd.DataFrame(a, columns=['one', 'two', 'three'])
    print("> df:\n%s\n" % df)
    print("> dtypes:\n%s" % df.dtypes) 
    div() 
    # df[['two', 'three']] = df[['two', 'three']].astype(float)
    df[['two']] = df[['two']].astype(float)
    print("> df:\n%s\n" % df)
    print("> dtypes:\n%s" % df.dtypes) 
    
    return

def t_select(): 
    """
    Memo
    ----
    1. df.iloc[i] returns the ith row of df. i does not refer to the index value, i is a 0-based index.
       In contrast, the attribute index is returning index values.
    """
    def show(df, prompt=''): 
        if prompt: 
            print('> %s\n%s\n' % (prompt, df))
        else: 
            print('\n%s\n' % df)
        return 
    div(message='Use case #1: select rows using row indices')
    df = pd.DataFrame({'BoolCol': [True, False, False, True, True]}, index=[10,20,30,40,50])
    idx = df[df['BoolCol'] == True].index.tolist()
    # select the rows 
    print df.loc[idx]
    
    div(message='Use case #2: select rows where columns match certain values.')
    df = pd.DataFrame({'A': 'foo bar foo bar foo bar foo foo'.split(),
                   'B': 'one one two three two two one three'.split(),
                   'C': np.arange(8), 'D': np.arange(8) * 2})
    show(df)
    show(df.loc[df['A'] == 'foo'])
    print('\nVS\n')
    criteria = {'A': 'foo'}
    show( match(df, criteria) )

    print('\n> for multiple values?')
    show(df.loc[df['B'].isin(['one','three'])])

    print('> another way')
    df = df.set_index(['B'])
    show(df.loc['one'])

    print('> multiple values ...')
    show(df.loc[df.index.isin(['one','two'])])

    return

def t_dtype2(**kargs): 
    def simplify(d): # e.g. 2009-08-31-13.00.00.123456
        d = d.split('-')
        return '-'.join(d[:3])

    import os, gzip, shutil 
    fname = 'fake.csv'
    base = os.path.join(ProjDir, 'test')
    fp = fpath = os.path.join(base, fname)

    buffer_path = 'data-lab/buffer'
    dpath = '/phi/proj/poc7002/bulk_training/data/cdr/lab/cerner/36271'
    fpath = '/phi/proj/poc7002/bulk_training/data/cdr/lab/cerner/36271/1994.gz'
    print('info> input path: %s' % fpath)
    assert os.path.exists(fpath)
    print('info> loading data from %s' % fpath)

    # sep='|', usecols=usecols, names=header, header=None, index_col=False, dtype=dtypes, error_bad_lines=error_bad_lines
    # header <- None to use default header, headr = [my header]
    # header = ['mrn', 'date', 'comp', 'parent', 'comp_code', 'value_type', 'num_value', 'char_value']
    header = ['mrn', 'date', 'comp', 'comp_code', 'value_type', 'num_value', 'char_value']
    # usecols = kargs.get('usecols', range(0, len(header)))
    usecols = [0, 1, 2, 4, 5, 6, 7]
    dtypes = {'mrn': 'int64'}

    
    # # reading gzipped file
    # fp = gzip.open(fpath)
    # # df = load_df(fp, sep='|', index_col=False, header=None, dtypes=dtypes, error_bad_lines=False, usecols=usecols, names=header)
    # df = pd.read_csv(fp, sep='|', usecols=usecols, names=header, 
    #                             header=None, index_col=False, dtype=dtypes, error_bad_lines=False) 
    df = pd.read_csv(fpath, sep='|', compression='gzip', usecols=usecols, names=header, 
                                header=None, index_col=False, dtype=dtypes, error_bad_lines=False)
    print df[:10]
    print('info> dim(df): %s | size of file: %f' % (str(df.shape), os.path.getsize(fpath)))
    print('-' * 100)
    
    cpath = os.path.join(buffer_path, 'temp.gz')
    with open(cpath, 'wb') as wfp:
        for fn in os.listdir(dpath):
            # print('combining %s' % fn)
            with open(os.path.join(dpath, fn), 'rb') as rfp:
                 shutil.copyfileobj(rfp, wfp)   # src -> dest
            # print('info> added %s => size: %f' % (fn, os.path.getsize(cpath))) 

    df = pd.read_csv(cpath, sep='|', compression='gzip', usecols=usecols, names=header, 
                                header=None, index_col=False, dtype=dtypes, error_bad_lines=False)
   
    print df[:10]
    print('info> dim(df): %s | size of file: %f' % (str(df.shape), os.path.getsize(cpath)))

    # df = df.replace(r'\s+', np.nan, regex=True)
    print('replacing empty strings with nan')
    # df['comp'].replace('\s+', '', inplace=True, regex=True)
    df['comp'].replace('\s+', np.nan, inplace=True, regex=True)
    print df[:10]

    print('drop those comp=NaN')
    df.dropna(subset=['comp'], inplace=True)
    print df[:10]

    print('sim date')
    df['date'] = df['date'].apply(simplify)

    print df[:10]

    return 

def t_input(**kargs):
    base = os.path.join(ProjDir, 'data-diag')
    fname = 'diag_098.0_ctrl.csv'
    fpath = os.path.join(base, fname)
    df = pd.read_csv(fpath, sep='|', index_col=False, header=0, error_bad_lines=True) 
    print df[:10]

def t_select2(**kargs):
    base = os.path.join(ProjDir, 'data-gold')
    fname = 'gold_candidates_neg_random_gh.csv'
    fpath = os.path.join(base, fname)
    print('info> input path: %s' % fpath)
    assert os.path.exists(fpath)
    print('info> loading data from %s' % fpath)

    dtype = {'icd9': str}
    df = pd.read_csv(fpath, sep='|', index_col=False, header=0, error_bad_lines=True, dtype=dtype) 
    cond_code = df['icd9']=='054.19' 
    cond_id = df['mrn']=='xxxxx'
    rows = df[cond_id & cond_id]
    print('info> rows:\n%s\n' % rows.to_string(index=False))


def test(): 
    # df = load_df(_file='cerner', is_data=True)
    # ref_df = load_df(_file='diag', is_data=True)
    # if not df is None: 
    #     spec(df)
    # if not ref_df is None: 
    #     from cdrLab import filterByDate2
    #     filterByDate2(df, ref_df, err=(-2*30, 1*30), _load=False, _save=True)
    # test_manipulate()
    # test_subsetting()
    # t_combine()
    
    # files = ['bt-100-0.csv', 'bt-100-1.csv', 'bt-100-2.csv', 'bt-100-3.csv', 'bt-100-4.csv']
    # files = ['bt-100-test-0.csv']
    # t_profile(path='/phi/proj/poc7002/bulk_training/dev', files=files)
    
    # t_typecast()

    # t_filter()

    ### reading csv or other input files 
    # t_input()

    ### data type
    # t_dtype()
    
    # what is the value for an empty string or space in .csv? 
    t_dtype2() 

    # t_select()

    # t_select2()

    return


if __name__ == "__main__":
    test() 

