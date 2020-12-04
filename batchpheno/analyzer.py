
#  File 
#  ----
#  analyzer.py 
#  
# 
#  Objective
#  ---------
#  Mainly to analyze (training) data produced by learner.py 
# 
#  1. Group similar features: 
#     a. document frequency 
#
#     b. TDIDF 
#
#     c. medcode type 
#
#  2. Analyze feature file
#     related: fparser.py 
#  
#
#  Features
#  ------- 
#  * generate feature description 
#      - genFeatureDescription()
#      - descriptionToCSV()
#         use qrymed to generate output first then call this 
#         routine to convert output to csv format.
# 
#  * cluster feature descriptions 
#      - naiveCluster*()
#
#  * feature descriptions 
#    - generate descr csv file from qrymed output
#       descriptionToCSV(keyword='cerner', prefix=configure.DirAnaDir)
#    - generate descr csv file from training set data
#       tsetToFeatureDescription
#
#  * analyze diagostic files 
#     - getICD9Codes() 
#        > check if ICD9 codes match with what's expected (e.g. all infections related?)
#  
#  Related
#  -------
#  * fparser.py   
#    a prototype of analyzer 
#  * learner.py 
#    cluster features are fed into learner.py via Group::augment() 
#
#  * labAnalyzer.py 
#  * medsAnalyzer.py 
#
import sys, os, csv, math, re
from scipy import interp
from scipy.stats import sem  # compute standard error
from pandas import DataFrame, Series
import pandas as pd 

from scipy.stats import sem  # compute standard error

try:
    import cPickle as pickle
except:
    import pickle

# text feature extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import pairwise_distances

# user-defined packages
import dfUtils
try: 
    from dfUtils import convert_dtypes, save_df, load_df
except: 
    print('save_df, load_df not supported.')

from learner import Group, Feature
from utils import div
import qrymed2
from predicate import isNumber
import configure 

DataRoot = configure.DataRoot
# DataExpRoot = configure.DataExpRoot
ProjDir = configure.ProjDir
FeatureDir = configure.FeatureDir
# LabName, LabTest = 'cerner'
DataExpRoot = os.path.join(ProjDir, 'experiment')   # os.path.join(DataExpRoot, 'cdr/lab')  
DataDir = 'data-learner'
DataExpDir = os.path.join(DataExpRoot, DataDir)

from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from optparse import OptionParser
# using R 
# from pandas import *
# from rpy2.robjects.packages import importr
# import rpy2.robjects as ro
# import pandas.rpy.common as com

# Candidate Text Features 
antibio_features = [u'acid', u'add-vantage', u'amikacin', u'amoxi-clav', u'amoxicillin', u'amoxicillin-clav', 
                       u'amoxicillin-clavulanic', u'amphotericin', u'ampicillin', u'ampicillin-sulba', 
                       u'ampicillin-sulbactam', u'approval', u'azithromycin', u'aztreonam', 
                    u'bacitracin', u'bag', u'benzathine', u'cap', u'caspofungin', u'cefazolin', 
                    u'cefepime', u'cefotaxime', u'cefoxitin', u'cefpodoxime', u'ceftazidime', u'ceftriaxone', 
                    u'cefuroxime', u'cephalexin', u'ciprofloxacin', u'clarithromycin', u'clear', 
                    u'clindamycin', u'clotrimazole', u'comp', u'compoun', u'compound', u'dno', u'doxorubicin', 
                    u'doxycycline', u'econazole', u'erythromycin', u'ext', u'extemp', u'fluconazole', u'gentamicin', 
                u'imipenem-cil', u'inh', u'inj', u'iso-osmo', u'iso-osmot', u'iso-osmotic', u'itraconazole', 
        u'ivpb', u'ivpb-adv', u'ivpb-pmix', u'ivpb-water', u'levofloxacin', u'linezolid', u'lipid', 
        u'liposome', u'loz', u'meropenem', u'mg-', u'miconazole', u'mitomycin', u'moxifloxacin', u'mupirocin', u'nasal', 
            u'neomycin-polymyxin-bacitracin', u'nystatin', u'oin', u'oint', u'opht', u'oral', u'oxacillin', 
            u'pen', u'piperacil-tazo', u'piperacil-tazobact', u'piperacil-tazobactam', u'piperacillin-tazobact', 
        u'piperacillin-tazobactam', u'pmix', u'polymyxin', u'potassium', u'premix', u'pwd', u'rifaximin', u'silver', 
        u'sirolimus', u'sod', u'sodium', u'sol', u'soln', u'sulf', u'sulfadiazine', u'sulfameth-trimethoprim',
        u'sulfamethoxazole-trimethoprim', u'sulfate', u'susp', u'syr', u'tab', u'tacrolimus', u'terbinafine', 
        u'ticarcil-clav', u'tobramycin', u'unit', u'vag', u'vancomycin', u'voriconazole', u'zinc']

microbio_features = []

# Some nice default configuration for plots
plt.rcParams['figure.figsize'] = 10, 7.5
plt.rcParams['axes.grid'] = True
plt.gray()


class Params(object): 
    """

    Note
    ----
    1. Stop words 
        * reasons of exclusion
            trivial stop words: the, and, *r, ... 

            unwanted cluster: rule, 
    """
    stop_words_map = \
            {'antibiotic': 
                      ['the', 'and', '*r', 'cerner', 'drug', 'approval', 'cap', 'clear', 'bag', 
                      'comp', 'inh', 'inj', 'ivpb', 'ivpb-adv', 'ivpb-pmix', 'ivpb-water', 'mg-', 'mg', 
                      'nasal', 'oint', 'oral', 'pwd', 'silver', 'unit', 'extemp', 'susp', 'soln', 'tab', 
                      'derived', 
                      ], 

             'microbiology': 
                   [ 'information', 
                     'previously', 'CPMC Organism Result', 
                    'cerner', 'lab', 'test', 'and', 'or',  'limitation', 'for', 'Hsv'
                    'cpmc laboratory test',
                    'me term',
                    'message',
                    'susceptibility type',
                    'microorganism', 
                    '--isolate',
                    'derived',
                    
                    'rule', 'obtained', 
                    'recovered', 'discontinued', 'significant', 'isolated', 'moderate', 
                    'processed', 'available', 'identification'
                    ]
            }

    stop_words = ['the', 'and', '*r', 'cerner', 'drug', 'approval', 'cap', 'clear', 'bag', 
                      'comp', 'inh', 'inj', 'ivpb', 'ivpb-adv', 'ivpb-pmix', 'ivpb-water', 'mg-', 'mg', 
                      'nasal', 'oint', 'oral', 'pwd', 'silver', 'unit', 'extemp', 'susp', 'soln', 'tab']
    mutual_exclusive_words = ['positive', 'negative', ]
    stop_phrases = ['cerner drug']

    desriptions = {'no': 'negation', 
                   '+': 'positive', 
                   '-': 'negative', 
                   'exclude': 'exclude', 
                   }
    type_unclassified = 'property'   # unclassified description type

    is_initialized = False

    file_exceptions = ['augment.*', ]  # if description file contains these keywords, then don't include them

    dervied_feature_format = '%s_%s'   # code, keyword(s)
    description_prefix = 'Derived'

    sep = '|'   # separator for feature description

    @staticmethod
    def getStopWords(filter_set='antibio.*'):
        def is_matched(gn):
            for pat in filter_set: 
                p = re.compile(r'%s' % pat)
                if p.search(gn): 
                    return True 
            return False

        if isinstance(filter_set, str): filter_set = [filter_set]
        assert hasattr(filter_set, '__iter__')
        for gn, words in Params.stop_words_map.items(): 
            if is_matched(gn): 
                print("getStopWords> matched group %s" % gn)
                return words 
        print("getStopWords> no group found matching with filter set: %s" % filter_set)
        return Params.stop_words

    @staticmethod
    def init(**kargs): 
        if Params.is_initialized: return
        Params.preprocess_stopwords()

        #[test]
        test_ = kargs.get('test_', 0)
        if test_: 
            div()
            pprint(Params.stop_words_map)
            pprint(Params.stop_words)

        Params.is_initialized = True
        return

    @staticmethod
    def preprocess_stopwords(): 
        import itertools
        def flatten(wlist):
            wlist_ = []
            for e in wlist: 
                if isinstance(e, list): 
                    for e_ in e: wlist_.append(e_)
                else: 
                    wlist_.append(e)
            return wlist_             

        for k, wlist in Params.stop_words_map.items(): 
            for i, w in enumerate(wlist): 
                ws = w.split()
                if len(ws) > 1: 
                    wlist[i] = [w_.lower() for w_ in ws]
                else: 
                    wlist[i] = w.lower() 
            # now flatten the list 
            Params.stop_words_map[k] = list(set(flatten(wlist))) # list(itertools.chain(*wlist))

        # for wlist in (Params.stop_words, Params.mutual_exclusive_words, ): 
        #     for i, w in enumerate(wlist): 
        #         ws = w.split()
        #         if len(ws) > 1: 
        #             wlist[i] = ws
        #     wlist = flatten(wlist)
        return 

def search(pattern='feature_description', prefix=None, ext='csv', filter_set=None, exceptions=None): 
    """
    Search files under a given directory (prefix) that match input pattern (pattern). 

    Arguments
    ---------
    pattern 
    prefix 
    ext 
    filter_set: partial regex patterns that candidate files have to match 
      e.g. ['fset', 'descr.*', '.*csv']
    exceptions: 

    Note 
    ----

    """
    import glob
    def fbasename(f): 
        fb = os.path.basename(f)
        mat = re.compile('(\w+)\.%s' % ext).match(fb)
        if mat: 
            return mat.group(1)
        return f
    def fullname(f): 
        return f + '.' + ext
    def soft_match(x, s):  # all patterns needs to match to be True
        if not s: return True
        if isinstance(s, str): s=[s]
        assert hasattr(s, '__iter__')
        for e in s: 
            p = re.compile(r'%s'%e)
            if not p.search(x): 
                return False 
        return True

    if prefix is None: prefix = DataExpDir
    if ext: 
        path = "%s/*.%s" % (prefix, ext)
    else: 
        path = "%s/*" % prefix

    print "search> path: %s" % path
    p = re.compile(r'%s'%pattern)

    candidates = [os.path.basename(f) for f in glob.glob(path) if p.search(f)]

    # file name should match filter_set but not exceptions 
    # if not filter_set: return candidates

    if exceptions is None: exceptions = Params.file_exceptions
    print "search> exceptions: %s" % exceptions
    candidates = [c for c in candidates if soft_match(c, filter_set) and not soft_match(c, exceptions)]

    print("search> candidate files: %s" % candidates)
            
    return candidates 

def isAugmentedFile(file_): 
    assert os.path.exists(file_), "Non-existent input file: %s" % file_
    fp = os.path.basename(file_)
    p = re.compile(r'augment')
    # ts = load_df(_file=self.get('file'), sep=',')  # check features, if mixed, then True
    if p.search(fp): 
        return True 
    return False

def load(file_=None, column=None, *args, **kargs): 
    """
    Load feature description files produced by learner. 

    Arguments
    ---------
    file_
    column

    <hidden>
    prefix: search directory 
    pattern: 
    filter_set: 
    type: return type: pair i.e. list of tuples or dict

    Note
    ----
    1. derived features: 
       medcodes are 'raw' features; any feature derived from aggregating 
       medcodes (e.g. cluster features) is considered a derived feature.

    Source 
    ------ 
    1. see features directory 
       e.g. features/blood.txt  

    """
    def to_pair(df, x, y):  # feature/medcode to description
        assert x in df.columns and y in df.columns
        if to_str: 
            return [(str(k), str(v)) for k, v in zip(df[x], df[y])]
        return zip(df[x], df[y])

    test_ = kargs.get('test', False)
    prefix = kargs.get('prefix', None)
    pattern = kargs.get('pattern', 'feature_description') # search files with this keyword
    filter_set = kargs.get('filter_set', 'antibio.*')  # further filter files with this regex
    to_str = kargs.get('to_str', True) # turn medcode/feature to string type 
    include_derived = kargs.get('include_derived', False)

    print("load> filter set: %s" % filter_set)
    if file_ is None: 
        # search candidate files under *prefix dir that match *pattern
        candidates = search(pattern=pattern, prefix=prefix, filter_set=filter_set)
        if test_: print "load> candidates: %s" % candidates
        if candidates: 
            file_ = candidates[0]
    assert file_ is not None, "analyzer::load> no candidate files found."
    prefix, fname = os.path.dirname(file_), os.path.basename(file_)
    path = os.path.join(DataExpDir, fname) if not prefix else file_
    assert os.path.exists(path), "load> invalid path: %s" % path
    
    # read csv 
    df = load_df(_file=path, root=prefix, sep=Feature.descr_sep) 
    if not include_derived:  # [1]
        df = df[df[Feature.descr_header[0]].map(lambda e: isNumber(e))]

    if not column: column = Feature.descr_header[1]
    
    # names = []
    # try: 
    #     names = df[column]
    # except: 
    #     raise ValueError, "analyze> Could not find column %s in dataframe" % column
    
    # for i, name in enumerate(names): 
    #     print("analyze> #%d => %s" % (i, name))
    # with open(path, 'rb') as csvfile:
    #     fr = csv.reader(csvfile, delimiter='|')
    #     for row in fr: # each row is a list
    #         if not row: continue
    #         fsets.append(row)
    # ftb = df.create_map()

    # ftb = to_pair(df, Feature.descr_header[0], Feature.descr_header[1])
    # for code, name in ftb: 
    #     print("%s => %s" % (code, name))
    type_ = kargs.get('type', 'pair')
    if type_.startswith('p'): 
        return to_pair(df, Feature.descr_header[0], Feature.descr_header[1])
    return dict(to_pair(df, Feature.descr_header[0], Feature.descr_header[1]))

def evalDescriptionTypes(file_=None, column=None, **kargs):
    """
    Evaluate the type/category of feature descriptions: 
    e.g. CPMC Organism Result
         Message
         ME Term: definition

    Output
    ------ 
    (ctod, dtb)
    where 
       ctod: a dict that maps from medcodes/features to their descriptions 
             (as those from qrymed -pnm <medcode>)
       dtb: maps from description types to medcodes/features

    Note
    ---- 
    1. Description type: 
       CPMC Organism Result
       Message
       ME Term: definition

    Related
    -------
    descriptionType()
    codeToType()

    """
    def inspect_type(x):
        x = x.lower() 
        xs = x.split(x)
        if x.startswith('rule out'): 
            return Params.desriptions['exclude']  # e.g. 39501: Rule Out Coxsackievirus
        if xs[0].lower() == 'no': 
            # if xs[1] == 'acid':  
            return Params.desriptions['no']
        if xs[0].lower() == 'positive': 
            return Params.desriptions['+'] 
        if xs[0].lower() == 'negative': 
            return Params.desriptions['-'] 
        return Params.type_unclassified

    if not kargs.has_key('filter_set'): kargs['filter_set'] = 'microbio.*'
    ftb = kargs.get('ftb', None)
    if ftb is None: ftb = load(file_, column, **kargs)  
    min_df = kargs.get('min_df', 1)
    stop_words = Params.getStopWords(kargs['filter_set']) if kargs.get('use_stop_words', True) else None    
    test_= kargs.get('test_', 1)

    # TfidfVectorizer, CountVectorizer
    vectorizer = CountVectorizer(min_df=min_df, stop_words=stop_words, token_pattern=r'[-a-zA-Z]{3,}')  # [1]
    doc = [name for _, name in ftb] 
    X = vectorizer.fit_transform(doc)  # [2] # names = [name for _, name in ftb]
    print "evalDescriptionTypes> dim X: %s, type: %s" % (str(X.shape), type(X))
    print("evalDescriptionTypes> number of docs/descriptions: %s" % len(doc))

    feature_names = vectorizer.get_feature_names() 
    if test_: div(message=feature_names, symbol='~')   

    ctod, ctod_short = {}, {}
    dtb = {Params.type_unclassified: []} 
    dX = X.toarray()
    pmsg = re.compile(r"(?P<type>\w+(\s+\w+)*):(?P<value>.*)")
    types = set([])
    for i, x in enumerate(dX):
        kws = [feature_names[j] for j, e in enumerate(x) if e == 1]
        if test_: print "%s => %s" % (doc[i],  kws)
        m = pmsg.match(doc[i].lower())
        typ = m.group('type') if m else Params.type_unclassified

        if typ in (Params.type_unclassified, ): 
            typ = inspect_type(doc[i])

        if test_: 
            ctod[ftb[i][0]] = (doc[i], kws, typ)
        else: 
            ctod[ftb[i][0]] = typ
        types.update([typ])
            
        if not dtb.has_key(typ): dtb[typ] = [] 
        if test_: 
            dtb[typ].append((ftb[i][0], kws))  # or use 'raw' description ftb[i][1]
        else: 
            dtb[typ].append(ftb[i][0])
    
    if test_ > 2: 
        div(); pprint(ctod)
        div(); pprint(dtb)
        
    return (types, ctod, dtb)
def descriptionType(file_=None, column=None, **kargs):
    types, x, y = evalDescriptionTypes(file_=file_, column=column, test_=0, **kargs)
    return types
def codeToType(file_=None, column=None, **kargs):
    types, x, y = evalDescriptionTypes(file_=file_, column=column, test_=0, **kargs)
    return x
def typeToCode(file_=None, column=None, **kargs): 
    types, x, y = evalDescriptionTypes(file_=file_, column=column, test_=0, **kargs)
    return y

def inspect(file_=None, column=None, *args, **kargs): 
    """
    searc
    Arguments
    --------- 
    1. 

    2. hidden

    Note
    ----
    1. CountVectorizer
         min_df: min doc freq 

       TfidfVectorizer
         Equivalent to CountVectorizer followed by TfidfTransformer.

    2. vectorizer.fit_transform() returns scipy sparse matrix

    3. pair-wise distances 
       <ref> 
         a. http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html
         b. http://scikit-learn.org/stable/modules/metrics.html
         c. http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise
    """
    ftb = load(file_, column, *args, **kargs)  # ftb: medcode feature to its description

    min_df = kargs.get('min_df', 1)

    # TfidfVectorizer, CountVectorizer
    vectorizer = TfidfVectorizer(min_df=min_df, stop_words=Params.stop_words, token_pattern=r'[-a-zA-Z]{3,}')  # [1]
    # vectorizer = CountVectorizer(min_df=min_df, stop_words=Params.stop_words, token_pattern=r'[-a-zA-Z]{3,}')  # [1]

    doc = [name for _, name in ftb] 
    X = vectorizer.fit_transform(doc)  # [2] # names = [name for _, name in ftb]
    print("analyze> dim X: %s, type: %s" % (str(X.shape), type(X)))
    print("analyze> number of docs/descriptions: %s" % len(doc))

    feature_names = vectorizer.get_feature_names()
    div(message=feature_names, symbol='~')

    # for i, x in enumerate(X):  # type of x is a scipy.sparse.csr.csr_matrix
    #     print "%s =>\n%s (type: %s)" % (feature_names[i], str(x), type(x))

    # convert sparse matrix to dense matrix
    div(message="Find word features for each document")
    dX = X.toarray()
    for i, x in enumerate(dX): 
        # if i % 10 == 0: 
           assert len(x) == len(feature_names)
           print "%s => %s" % (doc[i],  [feature_names[j] for j, e in enumerate(x) if e != 0]) #  [feature_names[e] for e in x if e == 1]

    # tokenize sentence
    analyze = vectorizer.build_analyzer()
    # d = "Cerner Drug: AMOXICILLIN SUSP 400 MG/5ML EXTEMP"
    # print 'inspect> description: %s => %s' % (d, analyze("Cerner Drug: AMOXICILLIN SUSP 400 MG/5ML EXTEMP"))
    for j, d in enumerate(doc): 
        print('inspect> description: %s => %s' % (d, analyze(d)))

    div(message="Compute pairwise similarity")
    # cluster features based on keywords
    s = pairwise_distances(dX, metric='jaccard') # scipy distance metrics do not support sparse matrices.
    print("analyze> s: %s" % s)

    # simialrity between desriptions
    # for i, x in enumerate(dX):
    #     print "%s"
    #     for j, y in enumerate(dX): 
    #         if i == j: continue 
    #         if jaccard_similarity_score(x, y) > 0: 
    #             print "%s ~ %s" % (doc[i], doc[j]) 
    return

def naiveCluster(file_=None, column=None, **kargs): 
    """

    Arguments
    ---------
    file_
    column 

    <hidden>  
    test: 0 if disabled, 
             if >=1: the higher, the more verbose
    filter_set: a string (of regex) or a list of strings (of regex)
    remove_empty:      

    Note 
    ---- 
    1. more general matching strategy
    2. cwmap: internal keyword rep. for each feature/medcode
    3. 
       a. if code x y not in same description type, can't be in same cluster 
       b. if wx, wy contains mutually exclusive keywords, then they cannot be in the same cluster 
         
    4. kx for different cx may not be unique

    Bug
    --- 
    1. unwanted clusters
       e.g. Cerner Drug: Linezolid Tab 600 mg *r*
            Cerner Drug: Linezolid Approval *r  => this is not a dosage
    2. ftb return from load() 
       mecode of type numpy interger
    3. CountVectorizer sometimes failed to fit_transform doc
         : empty vocabulary; perhaps the documents only contain stop words
         : this has to do with ftb{}? 
    """
    def sort_len(x): 
        assert hasattr(x, '__iter__')
        return [e[0] for e in sorted([(e, len(e)) for e in x], key=lambda e: e[1], reverse=True)]
    def bimatch(x, y): 
        px = re.compile(r'%s' % x)
        py = re.compile(r'%s' % y)
        if px.search(y) or py.search(x): 
            return True 
        return False
    def unimatch(x, y): 
        n, m = len(x), len(y)
        if n > m: 
            ref = re.compile(r'%s' % y) 
            if ref.search(x):  # search smaller string in longer one
                return True 
        else: 
            ref = re.compile(r'%s' % x)
            if ref.search(y): 
                return True  
        return False
    def update_clusters(i, group=None):
        e = ftb[i][0]
        if not clusters.has_key(e):  # 0:code, 1:descr 
            clusters[e] = [] 
        if group:  
            for j in group: 
                clusters[e].append(ftb[j][0])
        return
    def translate(): 
        ftb_ = dict(ftb)
        for k, v in clusters.items(): 
            print "%s" % ftb_[k]
            for e in v: 
                print "    %s" % ftb_[e]
        return 
    def size(): 
        s = set()
        for k, v in clusters.items(): 
            s.update([k])
            if v: 
                s.update(v)
        return len(s)
    def update_cwmap_test(i, j, u, v, doc=None, tdidf_features=None):
        c1, c2 = ftb[i][0], ftb[j][0]
        if not doc: 
            cwmap[c1], cwmap[c2] = u, v
        else: 
            if not tdidf_features: 
                cwmap[c1], cwmap[c2] = (u, doc[i]), (v, doc[j])
            else: 
                tdidf_features = dict(tdidf_features)
                cwmap[c1], cwmap[c2] = (u, doc[i], u in tdidf_features), \
                                       (v, doc[j], v in tdidf_features)
        return 
    def update_cwmap(clusters, cwmap):  # each cluster head has unique identifier in terms of keywords, etc.
        ctb = {}
        for h, members in clusters.items(): 
            kw = cwmap[h]
            if not ctb.has_key(kw): ctb[kw] = []
            ctb[kw].append(h) 
        for kw, heads in ctb.items(): 
            if len(heads) > 1: 
                for h in heads: 
                    cwmap[h] = Params.dervied_feature_format % (kw, h)
        return

    def is_mutual_exclusive(wx, wy): 
        wx_ = wy_ = None
        for w in Params.mutual_exclusive_words: 
            if w in wx: wx_ = w; break 
        if wx_ is None: return False
        for w in Params.mutual_exclusive_words: 
            if w in wy: wy_ = w; break 
        if wy_ is None: return False 
        return wx_ != wy_   # e.g. positive vs negative
    def type_check(ftb):
        import random
        n = random.randint(0, len(ftb)-1)
        i = 0 
        for k, v in ftb: 
            if i in (n-1, n, n+1,): 
               div(message="type of key %s: %s, type of value %s: %s" % (k, type(k), v, type(v)), symbol='~') # []
               break
            i+=1

    Params.init()
    clusters, cwmap, assigned = ({}, {}, set()) # [2]
    
    ftb = load(file_, column, **kargs)    # maps from code to names/descriptions
    ctb = codeToType(ftb=ftb)      # maps from code to description types

    test_= kargs.get('test', 0)
    min_df = kargs.get('min_df', 1)
    stop_words = Params.getStopWords(kargs.get('filter_set')) if kargs.get('use_stop_words', True) else None 
    if test_ > 2: type_check(ftb)
    print("naiveCluster> stop words: %s" % stop_words)
    doc = [name for _, name in ftb]
    # TfidfVectorizer, CountVectorizer
    vectorizer = TfidfVectorizer(min_df=min_df, stop_words=stop_words, token_pattern=r'[-a-zA-Z]{3,}')  # [1]

    try:  # [d3]
        X = vectorizer.fit_transform(doc)  # [2] # names = [name for _, name in ftb]
    except Exception: 
        div(message="naiveCluster> %s" % sys.exc_info()[1], symbol='~')
        vectorizer = TfidfVectorizer(min_df=min_df, stop_words=None, token_pattern=r'[-a-zA-Z]{3,}') 
        X = vectorizer.fit_transform(doc)
    print "naiveCluster> dim X: %s, type: %s" % (str(X.shape), type(X))
    print("naiveCluster> number of docs/descriptions: %s" % len(doc))

    feature_names = vectorizer.get_feature_names()
    tdidf_features = getTdIdfWeightedFeatures(ftb=ftb, **kargs) if test_ > 1 else None
    # print("naiveCluster> test level: %s | tdidf-weighted features:" % test_); div(); pprint(tdidf_features); div()
    
    print('naiveCluster> Number of word features: %d' % len(feature_names))
    if test_: 
        print('naiveCluster> Number of features: %d' % len(feature_names))
        div(message=feature_names, symbol='~')
        if test_ > 1: 
            print('naiveCluster> Number of tdidf-weighted features: %d' % len(tdidf_features))
            div(message=tdidf_features, symbol='~')

    # for i, x in enumerate(X):  # type of x is a scipy.sparse.csr.csr_matrix
    #     print "%s =>\n%s (type: %s)" % (feature_names[i], str(x), type(x))
    # convert sparse matrix to dense matrix
    is_loose = kargs.get('loose', True) 
    match = bimatch if is_loose else unimatch
    dX = X.toarray()
    format_derived_feature = Params.dervied_feature_format

    # cluster_kws = set()    # ideally, each cluster should be represented by unique keyword(s)
    for i, x in enumerate(dX): 
        assigned.update([i])
        cx = ftb[i][0]
        wx = set([feature_names[u] for u, e in enumerate(x) if e != 0])  # word features for x
        setx = sort_len(wx) 
        if not setx: update_clusters(i); continue
        kx = setx[0]
        # print "naiveCluster> %s => %s" % (doc[i], setx)
        group = []

        cwmap[cx] = kx # '%s_%s' % (cx, kx)  # maps from code to its keyword(s) [4]

        for j, y in enumerate(dX[i+1:]): 
            eid = i+j+1   # true index starting from 0th of dX
            if eid in assigned: continue
            cy = ftb[eid][0]
            wy = set([feature_names[v] for v, e in enumerate(y) if e != 0])  # word features for y
           
            if ctb[cx] != ctb[cy]: continue  # [3a]
            if is_mutual_exclusive(wx, wy): continue # [3b]

            # check keyword rep. to see if y and x belongs to the same cluster
            sety = sort_len(wy) # find longest keyword
            if sety: 
                ky = sety[0]    # [1]
                if match(kx, ky): 
                    if test_ > 2: update_cwmap_test(i, eid, kx, ky, doc, tdidf_features)
                    if test_: print "   + %s ~ %s | %s ~ %s" % (doc[i], doc[eid], kx, ky)
                    group.append(eid)
                    assigned.update([eid])
        update_clusters(i, group)
    assert len(assigned) == X.shape[0]

    if kargs.get('remove_empty', False): 
        size_ = len(clusters) 
        for k, v in clusters.items(): 
            if not v: 
                clusters.pop(k)
        print "naiveCluster> removed %d empty clusters" % len(clusters) 

    if test_: # or 'test' in args: 
        msg = "naiveCluster> size of cluster: %d" % size()
        div(message=msg)
        if test_ > 1: 
            div(); pprint(cwmap); div()
            translate()
    
    if kargs.get('include_cwmap', False): 
        update_cwmap(clusters, cwmap)
        return (clusters, cwmap)
 
    return clusters           

def naiveCluster2(file_=None, column=None, **kargs): 
    clusters = naiveCluster(*args, **kargs)
    return [[k].extend(v) for k, v in clusters.items()]

def naiveCluster3(file_=None, column=None, **kargs): 
    kargs['include_cwmap'] = True
    return naiveCluster(file_, column, **kargs)

def getDescription(w): 
    """

    Todo
    ----
    1. more generalized interface of extracting info 
       from dervied features
       class Cluster
    """
    import re
    p = re.compile(r'(?P<word>\w+)_(?P<code>\d+)')
    m = p.match(w) 
    content = m.group('word') if m else w
    return Params.description_prefix + ': ' + content 

def translate(clusters, reference=None, **kargs): 
    if reference is None: 
        reference = load(type='dict', **kargs)
    for i, cluster in enumerate(clusters): 
        print "cluster %s: %s" % (i, cluster) 
        for e in cluster: 
            print "        > %s" % e
    return 

def test_vectorize(file_=None, column=None, *args, **kargs):
    def sort_len(x): 
        assert hasattr(x, '__iter__')
        return [e[0] for e in sorted([(e, len(e)) for e in x], key=lambda e: e[1], reverse=True)]
    
    ftb = load(file_, column, *args, **kargs)
    min_df = kargs.get('min_df', 1)  
    vectorizer1 = CountVectorizer(min_df=min_df, token_pattern=r'[-a-zA-Z]{3,}')   # stop_words=Params.stop_words
    vectorizer2 = TfidfVectorizer(min_df=min_df, token_pattern=r'[-a-zA-Z]{3,}')   # stop_words=Params.stop_words,
    doc = [name for _, name in ftb] 
    X1 = vectorizer1.fit_transform(doc) 
    X2 = vectorizer2.fit_transform(doc) 
    assert X1.shape == X2.shape
    dX1 = X1.toarray()
    dX2 = X2.toarray()
    names1, names2 = vectorizer1.get_feature_names(), vectorizer2.get_feature_names()
    for i, x in enumerate(dX1): 
        # if i % 10 == 0: 
           assert len(x) == len(names1) == len(names2)
           print "%s => %s => %s" % (doc[i],  
                 sort_len([names1[j] for j, e in enumerate(x) if e == 1]), 
                 sort_len([names2[j] for j, e in enumerate(dX2[i]) if e != 0]) ) #  [feature_names[e] for e in x if e == 1]
    return 

def getTdIdfWeightedFeatures(file_=None, column=None, **kargs): 
    ftb = kargs.get('ftb', load(file_, column, **kargs)) # ftb: medcode -> description
    min_df = kargs.get('min_df', 1) 
    stop_words = Params.stop_words if kargs.get('use_stop_words', True) else None 
    vectorizer = TfidfVectorizer(min_df=min_df, stop_words=stop_words, token_pattern=r'[-a-zA-Z]{3,}') # stop_words=Params.stop_words
    doc = [name for _, name in ftb]
    X = vectorizer.fit_transform(doc) 
    idf = vectorizer.idf_
    ranking = zip(vectorizer.get_feature_names(), idf)
    ranking = sorted(ranking, key=lambda e: e[1], reverse=True)

    if kargs.get('test', 0): 
        div(); pprint(ranking); div()
        div(message="getTdIdfWeightedFeatures> num of features: %d" % len(ranking))
    return ranking

def test_tdidf(**kargs):
    file_, column = kargs.get('file_', None), kargs.get('column', None)
    ftb = load(file_, column, **kargs)
    min_df = kargs.get('min_df', 1) 
    stop_words = Params.stop_words if kargs.get('use_stop_words', True) else None 
    vectorizer = TfidfVectorizer(min_df=min_df, stop_words=stop_words, token_pattern=r'[-a-zA-Z]{3,}') # stop_words=Params.stop_words
    doc = [name for _, name in ftb]
    X = vectorizer.fit_transform(doc) 
    idf = vectorizer.idf_
    ranking = zip(vectorizer.get_feature_names(), idf)
    ranking = sorted(ranking, key=lambda e: e[1], reverse=True)
    pprint(ranking)
    div(message="test_tdidf> num of features: %d" % len(ranking))

    return

def test_similarity(): 
    """

    Note 
    ---- 
    1. Jaccard 
       the order counts? 
    """
    import numpy as np
    from sklearn.metrics import jaccard_similarity_score
    y_pred = [0, 2, 1, 3]
    y_true = [0, 1, 2, 3]
    print jaccard_similarity_score(y_true, y_pred)  
    print jaccard_similarity_score(y_true, y_pred, normalize=False)


def genFeatureDescription(keyword='microbio.*', fname=None, prefix=None, ext='csv', **kargs): 
    """
    Get feature descriptions for all features matching the keyword (without including
    their descendents).

    Related
    -------
     * genAllFeatureDescription(): same but include descendents


    Todo
    ----
    1. filter features by criteria

    """
    def extract_words(regex, sep='_'): 
        # find all words
        p = re.compile(r'\b[a-z]+\b', re.I)
        wlst = p.findall(regex)
        if wlst: 
            return sep.join(wlst)
        return None

    import sutils   # shell utility
    if prefix is None: prefix = FeatureDir
    if fname is None: fname = os.extsep.join((extract_words(keyword), ext))
    # test_ = kargs.get('test_', False)
    cmd = 'qrymed -find %s | qrymed -e -pnm' % keyword
    try: 
        fd = open(os.path.join(prefix, fname), 'w')
        fd.write(Params.sep.join(Feature.descr_header)+'\n')
        lines = sutils.execute(cmd)
        acc = 0
        for line in lines.split('\n'): 
            row = line.split()
            fd.write(row[0]+Params.sep+' '.join(row[1:])+'\n')
            acc += 1
    finally: 
        fd.close()

    return acc

# [todo] factor to tanalyzer.py
def tsetToFeatureDescription(file_, prefix=None, fname=None, **kargs): 
    """Given a training set, find all the features and their descriptions.
    """
    import predicate, sutils, dfUtils
    root, base = os.path.dirname(file_), os.path.basename(file_)
    if not root:
        root = prefix if prefix else os.path.abspath(os.curdir)
    df = dfUtils.load_df(_file=base, root=root, sep=',')
    # assert isinstance(df, DataFrame)
    if not fname: 
        fname = 'feature_description' + '_' + base
    cmd = 'qrymed -e -pnm %s'
    opath = os.path.join(root, fname)
    try: 
        fd = open(opath, 'w')
        fd.write(Params.sep.join(Feature.descr_header)+'\n')
        acc = 0
        for column in df.columns: 
            if not predicate.isNumber(column): continue 
            line = sutils.execute(cmd % column)
            row = line.split()
            fd.write(row[0]+Params.sep+' '.join(row[1:])+'\n')
            acc += 1
    finally: 
        fd.close()
    print("tsetToFeatureDescription> Generated file %s" % opath)
    return acc

def convertDescriptionToCSV(keyword='cerner', fname=None, prefix=None, ext=None, **kargs): 
    """
    Convert a qrymed output (in the format of code and description) to a csv format.

    Arguments
    ---------

    **kargs: 
      filter_set

    """
    import qrymed2
    candidates = [os.path.join(fname, prefix)]
    if keyword is not None: 
        filter_set = kargs.get('filter_set', None)
        candidates = search(pattern=keyword, prefix=prefix, ext=ext, filter_set=filter_set, exceptions='csv')
        print("test> candidates: %s" % candidates)
    for c in candidates: 
        qrymed2.descriptionToCSV(c, verbose=True)
    return
descriptionToCSV = convertDescriptionToCSV

def getMostFrequentFeatures(): 
    """Given a multirun file (where each run consists of a set of selected features, 
        find the most freqently-used features and their descriptions. 
    """
    pass

def genAllFeatureDescription(keyword='microbio.*', fname=None, prefix=None, ext='csv', **kargs): 
    """

    Todo
    ----
    1. avoid executing qrymed one by one
    """
    def extract_words(regex, sep='_'): 
        # find all words
        p = re.compile(r'\b[a-z]+\b', re.I)
        wlst = p.findall(regex)
        if wlst: 
            return sep.join(wlst)
        return None

    import sutils, qrymed2
    if prefix is None: prefix = FeatureDir
    if fname is None: fname = os.extsep.join((extract_words(keyword), ext))

    codes = qrymed2.filterByName(name=keyword, _load=True)
    print("genAllFeatureDescription> keyword %s has %d features including descendents" % (keyword, len(codes)))
    cmd = 'qrymed -pnm %s' 
    try: 
        fd = open(os.path.join(prefix, fname), 'w')
        fd.write(Params.sep.join(Feature.descr_header)+'\n')
        acc = 0
        for code in codes:
            doc = sutils.execute(cmd % code)   # slow [t1]
            fd.write(str(code)+Params.sep+doc+'\n')
            acc += 1
    finally: 
        fd.close()
    
    return acc

def test_description(**kargs): 
    labtests = ('fever', 'blood', 'antibiotic', 'microbiology')
    for labtest in labtests:
        n = genAllFeatureDescription(keyword=labtest)
        print "labtest %s has %d features" % (labtest, n)

def test_clustering(**kargs): 
    # files: 
    #   data-exp/mat/meds/erner_antibiotic_481_tset_mixed.csv
    # 
    from learner import Group, Feature
    import dfUtils
    file_ = kargs.get('file_', os.path.join(ProjDir, 'data-exp/cdr/lab/cerner/cerner_microbiology_481_tset_mixed.csv'))
    ts = load_df(_file=file_, sep=',')
    print "test_clustering> original data dim: %s" % str(ts.shape)
    # ts = Group.canonicalize(ts)
    ts = dfUtils.subset(ts, n_samples=1000)
    X, y = Group.transform(ts)
    print "test_clustering> dim X: %s, y: %s" % (str(X.shape), str(y.shape))
    X = add_noise(X)
    spectralCocluster(X, y, fname='cerner_microbiology_481_mixed')

    return

def add_noise(X, gamma=200):
    import random
    X = np.array(X)  # this will not cause side effect 
    m, n = X.shape
    for i in range(0,m): 
        for j in range(0,n):
            if X[i,j]==0: 
                X[i,j] = random.random()/(gamma+0.0)
    return X

def spectralCocluster(X, y=None, **kargs): 
    from matplotlib import pyplot as plt

    from sklearn.datasets import make_biclusters
    from sklearn.datasets import samples_generator as sg
    from sklearn.cluster.bicluster import SpectralCoclustering
    from sklearn.metrics import consensus_score

    plt.clf()

    data = X
    print("data dim %s, type: %s" % (data.shape, type(data)))
    print X[1, :]

    plt.matshow(data, cmap=plt.cm.Blues)
    plt.title("Original dataset")

    # data, row_idx, col_idx = sg._shuffle(data, random_state=0)
    plt.matshow(data, cmap=plt.cm.Blues)
    plt.title("Shuffled dataset")

    model = SpectralCoclustering(n_clusters=5, random_state=0)
    model.fit(data)
    # score = consensus_score(model.biclusters_,
    #                     (rows[:, row_idx], columns[:, col_idx]))

    # print("consensus score: {:.3f}".format(score))
    
    fit_data = data[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]

    plt.matshow(fit_data, cmap=plt.cm.Blues)
    plt.title("After biclustering; rearranged to show biclusters")

    if kargs.get('save_', False): 
        fname = kargs.get('file_', None)
        if fname is None: 
            fname = 'spectral_cocluster.png'
        else: 
            if fname.find('.png') < 0: fname += '.png'
        plt.savefig(os.path.join(DataExpDir, fname), bbox_inches='tight')
    else: 
        plt.show()
    
    return

def inspect_tset(**kargs): 
    """

    Files
    -----
    
    a. antibio
    * data-meds/cerner_antibiotic_481_tset_mixed.csv (searched by keyword)
    * data-meds/cerner/cerner_antibiotic_481_tset_mixed.csv (searched by selected medcodes only)

    b. microbio
    * data-lab/cerner/cerner_microbio_481_tset_mixed.csv

    c. blood 
    * data-lab/cerner/cerner_blood_481_tset_mixed.csv

    d. urine 
    * data-lab/cerner/cerner_urine_481_tset_mixed.csv

    """
    from utils import div
    from learner import Group, Feature
    from pprint import pprint
    from dfUtils import profile
    
    default = 'data-lab/cerner/cerner_blood_tset_mixed_infections_bt.csv' 
    fp = kargs.get('file_', default)
    root, fname = os.path.dirname(fp), os.path.basename(fp)
    if not root: root = ProjDir
    fp = os.path.join(root, fname)
    df = load_df(_file=fp, from_csv=True, sep=',')
    
    # 1. generate descr csv file from qrymed output
    # descriptionToCSV(keyword='cerner', prefix=configure.DirAnaDir)
    
    # 2. generate descr csv file from a tset file
    params = profile(df)
    div(); pprint(params); div()

    # df.columns is an index object
    df = Group.canonicalize(df)  # [w1][1]
    fg = Feature(df.columns)
    # print "> total feature set: %s" % fg.total()
    print("> number of features: %d =?= %d, type: %s" % (len(fg.total()), len(fg.active()), type(fg.total()) ))
    print("> number of columns:%s type: %s, examples: %s" % (len(df.columns), type(df.columns), df.columns[:10]))
    div()

    # if os.path.exists(fp): 
    #     root, base = os.path.dirname(fp), os.path.basename(fp)
    #     print('path %s is valid' % os.path.abspath(root))
    tsetToFeatureDescription(file_=fp)
    return 

### Analyzer for Diagnosis Table ### 

def match_icd9s(codeset, predicate, sep=',', verbose=False, verify_=False): 
    """
    Arguments
    ---------
    codeset: a set of strings of icd9 code sequence 
    verify_: if True, then if any sequence doesn't contain    
             at least a code that satisfies predicate, then 
             make an assertion or raise an exception. 
    """
    icd9targets = set()
    N = len(codeset); acc = 0
    for i, codestr in enumerate(codeset): 
        if not codestr: continue
        codelist = codestr.split(sep)
        if verbose: 
            if i % 100 == 0: 
                print("match_icd9s> codes: %s" % codelist)
        codes = [code for code in codelist if predicate(code)]
        if verify_: 
            # assert len(codes) > 0
            if len(codes) > 0: 
                print("match_icd9s> matched codes %s from %s" % (codes, codelist))
                acc += 1
        icd9targets.update(codes)
    if verify_: 
        print('match_icd9s> %d out of %d has a match. rate=%f' % (acc, N, (acc/(N+0.0))))
    return list(icd9targets)
def match_first_n_icd9s(codeset, predicate, n=1, sep=',', verbose=False, verify_=False, **kargs):
    icd9targets = set()
    N = len(codeset); acc = 0; n_null=n_null_str=n_intersect=0
    for i, codestr in enumerate(codeset): 
        if not codestr:
            n_null += 1 
            continue

        codelist = []
        try: 
            codelist = codestr.split(sep)
        except: 
            print('match_first_n_icd9s> codestr: %s' % codestr)
            n_null_str+=1

        if verbose: 
            if i % 500 == 0: 
                print("match_first_n_icd9s> codes: %s" % codelist)

        codes = []
        # print('match_first_n_icd9s> codelist: %s' % codelist)
        for i, code in enumerate(codelist): 
             if predicate(code): 
                codes.append(code)
                if len(codes) >= n: break
        if verify_: 
            # assert len(codes) > 0, "match_first_n_icd9s> no codes satify the predicate!"
            verify_codes = kargs.get('verify_codes', None)
            intersect = set(codes).intersection(set(verify_codes))
            if verify_codes is not None and len(intersect) > 0: 
                div(message='codes %s and %s has nonempty intersection: %s' % (codes, verify_codes, str(intersect)), symbol='%')
                n_intersect += 1
            if len(codes) > 0: 
                print("match_first_n_icd9s> matched codes %s from %s" % (codes, codelist))
                acc += 1
        icd9targets.update(codes)
    if verify_: 
        print('match_first_n_icd9s> %d out of %d has a match. rate=%f' % (acc, N, (acc/(N+0.0))))
        print('match_first_n_icd9s> %d records contain %s. rate=%f' % (n_intersect, str(verify_codes), (n_intersect/(N+0.0))))
        print('match_first_n_icd9s> %d null strings and %d nans' % (n_null_str, n_null))
    return list(icd9targets) 

# [todo] diagAnalyzer.py 
def getICD9Codes(file_, predicate=None, n_per_sequence=1, verbose=False, verify_=False, **kargs): 
    """
    Given a diag file, find all the icd9 codes that match 
    with predicate (e.g. isInfectiousParasitic())

    Arguments
    ---------
    n_per_sequence: the first n number of icd9 codes that (i.e. icd9 codes in a patient's record)
                    satisfies the predicate 
                    set to None to match all

    Todo
    ----
    1. Check strategy pattern to avoid unnecessary conditions in loop
    """
    import icd9utils
    # import configure
    # root = configure.DiagDataExpRoot
    assert os.path.exists(file_), "file %s does not exist." % file_
    print('getICD9Codes> input diag file: %s' % file_)
    df = load_df(_file=file_, from_csv=True, sep='|')
    codeset = None
    try: 
        codeset = df['icd9_code'].values
    except KeyError: 
        print('getICD9Codes> Invalid dataframe from %s' % f)
        print('getICD9Codes> header: %s' % [c for c in df.columns]) 
    if not predicate: 
        predicate = icd9utils.isInfectiousParasitic
    assert hasattr(predicate, '__call__')
    if n_per_sequence is None: 
        return match_icd9s(codeset, predicate=predicate, verbose=verbose, verify_=verify_)
    return match_first_n_icd9s(codeset, predicate=predicate, n=n_per_sequence, verbose=verbose, 
               verify_=verify_, **kargs)


def hasCode(codes, targets=None, predicate=None, min_n_match=None, verbose=False, sep=','):
    """
    Predicate for detecting occurrences of targets (e.g. unwanted 
    ICD9 codes) in codes (e.g. codes appeared in a patient's record). 

    Arguments
    ---------
    targets: a single code or a list of codes to be checked for their existence in codes
    min_n_match: the minimum number of codes in targets that occur in codes 
                 to consider as being True

    Related
    -------
    """
    if not targets: return True
    if not codes: return False
    if not hasattr(targets, '__iter__'): targets = [targets, ] 
    if isinstance(codes, str): 
        codes = codes.split(sep) 
        codes = [code.replace(" ","") for code in codes] 
    assert hasattr(codes, '__iter__'), "hasCode> codes after conversion: %s" % codes

    s = set(codes).intersection(set(str(t) for t in targets))
    if min_n_match is None: 
        if s: return True
    else: 
        if len(s) >= min_n_match: 
            return True
    return False

# [todo] diagAnalyzer.py 
def prune(file_, targets=None, predicate=None, min_n_match=None, verbose=False, sep=','):
    """
    Prune unwanted entries (in a csv file) that do not satisfy the 
    given predicate (e.g. create a data source for patients without infection-related
    icd9 codes from a file where some of the entries have mentions of infection-related
    codes)

    Memo
    ----
    1. check x=float('nan') with math.isnan(x)  but cannot work with unimplemented types such as 
       empty string
    """
    def is_null(e): 
        return (e is None) or (not e) or isNan(e) or np.isnan(e)
    def isNaN(e): 
        return e != e # only works for python 2.5+

    from utils import file_path
    import predicate
    fp = file_path(file_, default_root=os.path.abspath('data-diag'))  # file_ had better be the full path
    df = load_df(_file=file_, from_csv=True, sep='|')
    if verbose: div(message='Before pruning df: %s' % str(df.shape), symbol='*')
    df = df[df['icd9_code'].map(lambda e: predicate.isNull(e) or not hasCode(e, targets))]
    if verbose: div(message='After pruning df: %s' % str(df.shape), symbol='*')
    save_df(df, _file=fp, to_csv=True)
    return df

# [todo] learnerAnalyzer.py 
def displayModel(file_, identifier=None, fname=None, in_sep='|', out_sep=','):
    """
    Given a learner's output file with features and 
    their values, output a more human readable format.

    Note
    ----
    1. it's important to delete the reference to reader1 when you are finished with it; 
       otherwise tee will have to store all the rows in memory in case you ever call next(reader1) again
    """
    from utils import file_path
    import itertools 
    from pprint import pprint
    # automatic detection of learner description file
    # identifier=None, fname=None):
    #     if fname is None: 
    #         fname = 'stacking_description' + '_' + identifier + '.%s' % ext
        
    #     sep = '|'
    #     fpath = os.path.join(DataExpDir, fname)
    try: 
        fp = open(file_path(file_), 'rb')
        reader1, reader2 = itertools.tee(csv.reader(fp, delimiter=in_sep))
        # reader=csv.reader(f,delimiter=d)
        # ncol=len(next(reader)) # Read first line and count columns
        # f.seek(0) 
        header = next(reader1)
        ncol = len(header)
        # print('displayModel> ncol: %d' % ncol)
        ctab = {} # dict.fromkeys(header, None)
        del reader1  # [1]
        header = next(reader2)
        # print "header? %s" % header
        for i, row in enumerate(reader2):
            # print("[%d] row=%s" % (i, row))
            for j in range(ncol):
                if not ctab.has_key(header[j]): 
                    ctab[header[j]] = [] 
                ctab[header[j]].append(row[j])

        # [todo] gradual display 
        div(message="Showing features and their coefficients ...")
        
        # det the ordering
        # header.sort()
        atab = []
        for k, vec in ctab.items(): 
            atab.append( (k, np.mean([float(e) for e in vec])) )
        # print atab
        sorted_header = sorted(atab, key=lambda x:x[1], reverse=True)
        # print sorted_header
        for h, v in sorted_header: 
            print("%s:%s(avg) < %s" % (h, v, out_sep.join(ctab[h])))
        # div(); pprint(ctab); div()

    finally: 
        fp.close()
        del reader2 
        
    return        

def test(): 
    """

    Note
    ---- 
    1. Could be used to group microbiology features

    Log
    ---
    1. why 'acid' is a feature in antibio case?
    """
    # <key>
    filter_set = ['microbio.*', 'full']
    # description types, used particularly for microbiology lab tests
    Params.init()

    # <test>

    fp = os.path.join(FeatureDir, 'fever.txt')
    # inspect(file_=fp, include_derived=False)
    # test_description(keyword='fever')
    # test_vectorize(filter_set=['microbio.*', ])
    # test_tdidf(use_stop_words=True)
    
    # test_similarity()

    # x, y = evalDescriptionTypes(test_=False)  # [1]
    # pprint(x)
    # types = descriptionType(filter_set=filter_set); pprint(types)
    # code_type = codeToType(filter_set=filter_set); pprint(code_type)
    # type_code = typeToCode(filter_set=filter_set); pprint(type_code)

    # clusters, cwmap = naiveCluster3(test=1, filter_set=filter_set, remove_empty=True, to_str=True)  # filter_set='microbio.*'
    # div(); pprint(cwmap)
    # ctb = {}
    # for h, members in clusters.items(): 
    #     kw = cwmap[h]
    #     if not ctb.has_key(kw): ctb[kw] = []
    #     ctb[kw].append(h)
    # div(); pprint(ctb)
    # translate medcode representation to string representation
    # translate(clusters)
    

    # <key> get TDIDF-weigthed features 
    # getTdIdfWeightedFeatures(filter_set='microbio.*', test=True)
    

    # clustering 
    # test_clustering()

    # file maniputation (qrymed output to csv file)
    f = './data-lab/cerner/cerner_blood_tset_mixed_infections_bt.csv'
    f = './data-lab/cerner/cerner_microbio_tset_mixed_infections_bt.csv'
    f = './data-lab/cerner/cerner_microbio_008.45_tset_bt_100.csv'
    
    # f = './data-meds/cerner/cerner_antibio_070.3_tset_bt_100.csv'
    inspect_tset(file_=f)


    # diag
    # import diagReader
    # code = '038.8'
    # # f = './data-diag/diag_049.1_100-infections-as-1.csv'
    # f = './data-diag/diag_%s_ctrl.csv' % code 
    # # f = './data-diag/diag_no-infections-as-0-5000.csv'
    # icd9targets = getICD9Codes(f, verbose=False, verify_=True, verify_codes=[code, ])
    # print('test> found %d icd9 codes.\ntest> icd9targets:\n%s' % (len(icd9targets), icd9targets))
    # code_ = diagReader.show_codes(os.path.basename(f)); div(message='target code: %s' % code_, symbol='*')
    # assert not code_ in set(icd9targets)

    # diag pruning 
    # cfmap = diagReader.show_files(ctrl=True, path=os.path.abspath('data-diag'))
    # for code, file_ in cfmap.items(): 
    #     div(message='Prunning data for code %s' % code)
    #     prune(file_, targets=code, min_n_match=None, verbose=True, sep=',')
    # diagReader.show(verify_=True, recover_=True)

    # learner 
    f = 'data-learner/stacking_description_stack_ensemble_mixed-infections-bt.csv'
    # learner 
    # displayModel(file_=f)

    return


if __name__ == "__main__": 
    test()



















