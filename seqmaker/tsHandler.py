# encoding: utf-8

import seqConfig, seqparams
from tset import TSet
from tdoc import TDoc
from pandas import DataFrame, Series 
import pandas as pd

import numpy as np
import os, sys, re, random, time
import collections

# wrapper class: training set handler 
class tsHandler(object):  # <- seqparams, tset.TSet

    # default parameter values
    is_augmented = False 
    cohort = 'CKD'
    seq_ptype = ctype = 'regular'
    d2v = d2v_method = 'pv-dm2'  # pv-dm, bow
    is_simplified = False
    meta = 'D'  # training set file descriptor
    meta_model = None
    dir_type = 'combined'

    # class state
    is_configured = False
    N_train_max = N_max = 10000  # max training examples
    N_test_max = 5000

    @staticmethod
    def init(**kargs):   
        return tsHandler.config(**kargs)
    @staticmethod
    def config(cohort, seq_ptype='regular', d2v_method=None, simplify_code=False, 
            is_augmented=False, dir_type='combined', meta=None, meta_model=None, **kargs):
        # print('tsHandler.is_configured=? %s' % tsHandler.is_configured)
        # assert tsHandler.is_configured == False, "first call must be False"
        if not tsHandler.is_configured: 
            if d2v_method is None: d2v_method = vector.D2V.d2v_method 
            user_file_descriptor = meta
            tsHandler.cohort = cohort
            tsHandler.ctype = tsHandler.seq_ptype = seqparams.normalize_ctype(seq_ptype)
            tsHandler.d2v = d2v_method 
            tsHandler.is_simplified = simplify_code
            tsHandler.is_augmented = is_augmented
            tsHandler.dir_type = dir_type
            tsHandler.meta = meta if meta is not None else tsHandler.get_tset_id() # use training set ID as default
            tsHandler.meta_model = meta_model if meta_model is not None else tsHandler.get_model_id(kargs.get('y', [])) 
            
            print('config> d2v: %s, user descriptor (model, tset, mcs): %s' % (d2v_method, user_file_descriptor))
            print('        cohort: %s, ctype: %s' % (tsHandler.cohort, tsHandler.ctype))
            print('            + augmented? %s, simplified? %s dir_type=%s' % \
                (tsHandler.is_augmented, tsHandler.is_simplified, tsHandler.dir_type))
            tsHandler.is_configured = True
        return
    @staticmethod
    def isConfigured(): 
        return tsHandler.is_configured
    @staticmethod
    def is_sparse(): 
        return True if tsHandler.d2v.startswith(('bow', 'bag', 'aph', )) else False
    @staticmethod
    def load_tset(index=0, meta=None, sparse=False):
        tSparse = sparse # kargs.get('sparse', False)
        if tSparse: 
            return tsHandler.loadSparse(index=index, meta=meta)   # (X, y)
        return tsHandler.load(index=index, meta=meta) 
    @staticmethod
    def load(index=0, meta=None):  # cohort, seq_ptype, d2v_method (focused_labels, label_map)
        # this should conform to the protocal in makeTSetCombined()

        # if sparse: 
        #     X, y = tsHandler.loadSparse(index=index, meta=meta)
        #     return 

        # ts = TSet.load(cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype, 
        #         dir_type='combined', index=0) # [params] index
        ts = TSet.load(cohort=tsHandler.cohort, 
                       d2v_method=tsHandler.d2v, 
                       seq_ptype=tsHandler.ctype, 
                       index=index,  # CV index (if no CV, then 0 by default)
                       dir_type=tsHandler.dir_type, 
                       suffix=tsHandler.meta if not meta else meta) 
        assert ts is not None and not ts.empty  
        tsHandler.profile(ts)
        return ts
    @staticmethod
    def load_chunk(index=0, meta=None, chunksize=1000): 
        for tsi in TSet.loadChunks(cohort=tsHandler.cohort, 
                           d2v_method=tsHandler.d2v, 
                           seq_ptype=tsHandler.ctype, 
                           index=index,  # CV index (if no CV, then 0 by default)
                           dir_type=tsHandler.dir_type, 
                           suffix=tsHandler.meta if not meta else meta, 
                           chunksize=chunksize): 

            assert tsi is not None and not tsi.empty  
            tsHandler.profile(tsi)
            yield tsi

    @staticmethod
    def load_from(index=0, meta=None, verify_=True):
        fpath = TSet.loadFrom(cohort=tsHandler.cohort, 
                            d2v_method=tsHandler.d2v, 
                            seq_ptype=tsHandler.ctype, 
                            index=index,  # CV index (if no CV, then 0 by default)
                            dir_type=tsHandler.dir_type, 
                            suffix=tsHandler.meta if not meta else meta) 
        if verify_: 
            assert os.path.exists(fpath), "Invalid training set path:\n%s\n" % fpath
            print('(load_from) %s' % fpath)
        return fpath

    @staticmethod
    def loadSparse(index=0, meta=None):
        # load sparse matrix
        X, y = TSet.loadSparse(cohort=tsHandler.cohort, 
                            d2v_method=tsHandler.d2v, 
                            seq_ptype=tsHandler.ctype, 
                            index=index,  # CV index (if no CV, then 0 by default)
                            dir_type=tsHandler.dir_type, 
                            suffix=tsHandler.meta if not meta else meta)  
        if X is not None and y is not None: assert X.shape[0] == len(y)
        return (X, y)
    @staticmethod
    def save_tset(**kargs):
        return tsHandler.save(**kargs) 
    @staticmethod
    def save(X, y, index=0, docIDs=[], meta=None, sparse=False, shuffle_=True, verify_=False): # [params] (X_train, X_test, y_train, y_test)
        # can also use 'tags' (in multilabel format)
        print('tsHandler> save document vectors (cv=%d), sparse? %s ...' % (index, sparse))  # e.g. tpheno/seqmaker/data/CKD/train/tset-IDregular-pv-dm2-GCKD.csv
        # suffix=kargs.get('tset_id', kargs.get('seq_ptype', None))
        ts = None
        if not sparse: 
            ts = TSet.toCSV(X, y=y, save_=True, index=index,   
                               docIDs=docIDs,     # if a doc is segmented, then it's useful to know from which documents each segment came from
                               outputdir=tsHandler.get_tset_dir(),  # depends on cohort, dir_type
                            
                               d2v_method=tsHandler.d2v, 
                               cohort=tsHandler.cohort, 
                               seq_ptype=tsHandler.ctype, 
                               shuffle_=shuffle_, 
                               verify_path=verify_,     # set to True to bypass actuall saving if file already exists
                               suffix=tsHandler.meta if not meta else meta) # [params] labels, if not given => all positive (1)
        else: 
            TSet.saveSparseMatrix(X, y=y,
                    docIDs=docIDs, 
                    outputdir=tsHandler.get_tset_dir(), 
                    d2v_method=tsHandler.d2v, 
                               cohort=tsHandler.cohort, 
                               seq_ptype=tsHandler.ctype, 
                               suffix=tsHandler.meta if not meta else meta)
            # saves two files .npz (X) and .csv (y)
        return ts # or None
    @staticmethod
    def saveSparse(): 
        pass 

    @staticmethod
    def load_mcs(index=None, meta=None, inputdir=None, ifiles=[], inputfile=None): # use processDocuments()
        """
        Each training set should come with a document source file, which is derived from the source file. 

        Params
        ------
        meta: user-defined file ID

        Example
        -------
        1.  a. source file is derived directly from the OHASI DB; e.g. 

                tpheno/CKD/condition_drug_labeled_seq-CKD.csv
                    pattern: tpheno/<cohort>/condition_drug_labeled_seq-<cohort>.csv

            b.  derived mcs file is derived from the source (upon segmenting, simplifying, slicing operations, etc) and 
                it is from the derived mcs file that a d2v-model-based training set file is generated. 

                dervied mcs file is usually kept under a separate directory from the source is prefixed by 'mcs'; e.g. 

                    tpheno/seqmaker/data/CKD/combined/mcs-n0-IDregular-pv-dm2-D-GCKD.csv
                    
        """
        import TSet
        ### load + transfomr + (ensure that labeled_seq exists)
        
        # this is the MCS file from which the training set is made (not the source or orginal)
        # src_dir = tsHandler.get_tset_dir() if inputdir is None else inputdir  
        # assert os.path.exists(src_dir), "Invalid input dir: %s" % src_dir
        # ctype = tsHandler.ctype # kargs.get('seq_ptype', 'regular')
        
        # # inputfile takes precedence over ifiles
        # if inputfile is None: 
        #     inputfile = TSet.getName(cohort=tsHandler.cohort, d2v_method=tsHandler.d2v_method, index=index, 
        #                                 seq_ptype=ctype, suffix=tsHandler.meta if not meta else meta)
        # ipath = os.path.join(src_dir, inputfile)
        # assert os.path.exists(ipath), "Invaid source path: %s" % ipath
        # ifiles = [inputfile, ]

        # D, L, T = processDocuments(cohort=tsHandler.cohort, seq_ptype=ctype, 
        #             inputdir=src_dir, 
        #             ifiles=ifiles,
        #             meta=meta, 
                    
        #             # document-wise filtering 
        #             policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  
        #             min_ncodes=0,  # retain only documents with at least n codes

        #             # content modification
        #             # predicate=kargs.get('predicate', None), # reserved for segment_docs()
        #             simplify_code=tsHandler.is_simplified, 

        #             source_type='default', 
        #             create_labeled_docs=False)  # [params] composition
        ret = TDoc.load(inputdir=tsHandler.get_tset_dir(), 
                            inputfile=inputfile, 
                            index=index,   # index: k in k-fold CV or a trial index
                            
                            d2v_method=tsHandler.d2v, 
                            cohort=tsHandler.cohort, 
                            seq_ptype=tsHandler.ctype, 
                            suffix=tsHandler.meta if not meta else meta) # [params] labels, if not given => all positive (1)
        D, L, T = [], [], []
        if ret: 
            D, L, T = ret['sequence'], ret['label'], ret['timestamp']
        print('load_mcs> nD: %d | cohort=%s, ctype=%s, labeled? %s, simplified? %s' % \
            (len(D), tsHandler.cohort, ctype, tsHandler.is_labeled_data(L), tsHandler.is_simplified))
        return (D, L, T)
    @staticmethod
    def save_mcs(D, T=[], L=[], index=None, docIDs=[], meta=None, sampledIDs=[]):
        """
        
        Params
        ------
        meta: user-defined file ID
        index: used mainly for cross validation; set to None to omit index

        """
        from tdoc import TDoc
        # can also use 'tags' (in multilabel format)
        print('tsHandler> save transformed documents parallel to tset (cv=%d) ...' % index)  # e.g. tpheno/seqmaker/data/CKD/train/tset-IDregular-pv-dm2-GCKD.csv
        # suffix=kargs.get('tset_id', kargs.get('seq_ptype', None))
        
        # this should basically share the same file naming parameters as TSet.toCSV()
        tsDoc = TDoc.toCSV(D, T=T, L=L, save_=True, index=index,   # index: k in k-fold CV or a trial index
                            docIDs=docIDs,   # if a doc is segmented, then it's useful to know from which documents each segment came from
                            outputdir=tsHandler.get_tset_dir(),  # depends on cohort, dir_type
                            outputfile=None, # use default; i.e. auto naming
                            
                            d2v_method=tsHandler.d2v, 
                            cohort=tsHandler.cohort, 
                            seq_ptype=tsHandler.ctype, 
                            suffix=tsHandler.meta if not meta else meta) # [params] labels, if not given => all positive (1)
        
        if sampledIDs: 
            header = ['source_index', ]
            adict = {h: [] for h in header}
            for i in range(len(D)): 
                adict['source_index'].append(sampledIDs[i])
            df = DataFrame(adict, columns=header)
            TSet.saveDataFrame(df, 
                outputdir=tsHandler.get_tset_dir(),  # depends on cohort, dir_type
                index=index, 
                d2v_method=tsHandler.d2v, cohort=tsHandler.cohort, seq_ptype=tsHandler.ctype,
                suffix=tsHandler.meta if not meta else meta)

        return tsDoc # or None
    @staticmethod
    def profile(ts): 
        from tset import TSet
        col_target = TSet.target_field
        Lv = ts[col_target].unique()
        sizes = {}

        # sizes = collections.Counter(ts[col_target].values)
        for label in Lv: 
            sizes[label] = ts.loc[ts[col_target] == label].shape[0]
        print('tsHandler.profile> Found %d unique labels ...' % len(Lv))
        for label, n in sizes.items(): 
            print('  + label=%s => N=%d' % (label, n))
        return  
    @staticmethod
    def profile2(X, y): 
        sizes = collections.Counter(y)
        for label, n in sizes.items(): 
            print('  + label=%s => N=%d' % (label, n))
        return 
    @staticmethod
    def is_labeled_data(lx): 
        # if lx is None: return False
        nL = len(set(lx))
        if nL <= 1: 
            return False 
        return True
    @staticmethod     # [design] this can be a friend function or TSet's class method
    def get_tset_id(ts=None, y=[]): # (ts, y, include_augmetned)
        """

        Memo
        ----
        1. A typical training set file looks like: 
           tset-n0-IDregular-pv-dm2-A-GCKD.csv

           full path: tpheno/seqmaker/data/<cohort>/combined/tset-n0-IDregular-pv-dm2-A-GCKD.csv
        """
        meta = 'D'  # default (not distinguishing labeled (L) or unlabeled (U))
        nL = 0
        if ts is not None:  
            col_target = TSet.target_field
            nL = len(ts[col_target].unique())
            meta = 'U' if nL <= 1 else 'L' # labeled or unlabeled?
        elif y: 
            nL = len(set(y))
            meta = 'U' if nL <= 1 else 'L' # labeled or unlabeled?
        if tsHandler.is_augmented: meta = 'A'  # augmented data included?
        # if kargs.get('simplify_code', False): meta = '%s%s' % (meta, 'S')
        return meta  
    @staticmethod
    def get_tset_dir(): 
        from tset import TSet
        tsetPath = TSet.getPath(cohort=tsHandler.cohort, dir_type=tsHandler.dir_type)  # ./data/<cohort>/
        print('tsHandler> training set dir: %s' % tsetPath)
        return tsetPath 

    @staticmethod
    def get_model_id(y=[]): # <- L
        """
        Depends on training set property (get_tset_id()), which depends on labels (y or L)

        Memo
        ----
        1. A typical model file looks like: 
           

           Note that the suffix naming follows the convention of the ID field of the training set file name
                tset-n0-IDregular-pv-dm2-A-GCKD.csv => ID string: regular-pv-dm2-A-GCKD
        """    
        # add to model file to distinguish different d2v models in addition to its parameter-specific identifier
        # ID: {'U', 'L', 'A'} + ctype + cohort
        #     unlabeld (U), labeled (L) or augmented (A) (i.e. plus unlabeled)
        ctype = tsHandler.ctype # seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular'))
        meta = '%s-%s' % (ctype, tsHandler.d2v)
        
        tset_type = tsHandler.get_tset_id(y=y)  # property of the training data ('D'/default, 'L'/labeled, 'U'/unlabeled, 'A'/augmetned)
        meta = '%s-%s' % (meta, tset_type)
        
        # cohort_name = kargs.get('cohort', '?')
        if not tsHandler.cohort in ('?', 'n/a', None): meta = '%s-G%s' % (meta, tsHandler.cohort)
        if len(meta) == 0: meta = 'generic' # if no ID info is given, then it is generic test model
        return meta

    @staticmethod
    def get_model_dir():  # d2v model
        # (!!!) this could have been included in vector.D2V but do not want D2V to have to pass in 'cohort'
        #       the directory in which model is kept
        dir_type = 'model'
        modelPath = TSet.getPath(cohort=tsHandler.cohort, dir_type=dir_type, create_dir=True)  # ./data/<cohort>/model
        print('tsHandler> model dir: %s' % modelPath)
        return modelPath
    @staticmethod
    def get_nns_model_dir(): 
        dir_type = 'nns_model'
        modelPath = TSet.getPath(cohort=tsHandler.cohort, dir_type=dir_type, create_dir=True)  # ./data/<cohort>/model
        print('tsHandler> model dir: %s' % modelPath)
        return modelPath 

    @staticmethod
    def isBig(X):
        return X.shape[0] > tsHandler.N_max 

### end class tsHandler

def loadTSetCombined(**kargs): # used only for backward compatibility; combined: combining data splits
    return loadTSet(**kargs)  
def loadTSet(**kargs): 
    """
    Load training data. Similar to loadTSetSplit() but assume no separation between train and test splits. 

    Params
    ------
    dir_type: 
       {train, test}
       combined 
       {cv_train, cv_test}
    index: an integer in the file name used to distinguish data from different CV partitions. 

    w2v_method, d2v_method
    seq_ptype
    read_mode
    cohort

    Related
    -------
    1. see _TSetHandler

    """
    def config(): # use sysConfig() instead
        tsHandler.config(cohort=kargs.get('cohort', 'CKD'), 
            seq_ptype=kargs.get('seq_ptype', 'regular'), 
            d2v_method=kargs.get('d2v_method', vector.D2V.d2v_method), 
            is_augmented=kargs.get('is_augmented', False), 

            meta=kargs.get('meta', None), 
            simplify_code=kargs.get('simplify_code', False), dir_type='combined')
        return 
    def check_tset(ts): 
        target_field = TSet.target_field
        n_classes = 1 
        no_throw = kargs.get('no_throw', True)
        if ts is not None and not ts.empty: 
            # target_field in ts.columns # o.w. assuming that only 1 class 
            n_classes = len(ts[target_field].unique())  
            print('loadTSet> number of classes: %d' % n_classes)
        else:
            msg = 'loadTSet> Warning: No data found (cohort=%s)' % kargs.get('cohort', 'CKD')
            if no_throw: 
                print msg 
            else: 
                raise ValueError, msg
        return n_classes      
    def get_tset_id(ts=None): # L <-
        return tsHandler.get_tset_id(ts) 
    def profile(ts): 
        return tsHandler.profile(ts)   
    def get_tset_dir(): 
        return tsHandler.get_tset_dir()
    def customize_tset(ts): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Control"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            ts = focus(ts, labels=focused_classes, other_label='Control')
        lmap = kargs.get('label_map', seqparams.System.label_map)
        if lmap: # relabel so that classes only retain the main CKD stages
            print('(customize_tset) Modifying training data ...\n> Prior to re-labeling ...')
            profile(ts)
            ts = merge(ts, lmap)
            print('... After re-labeling ...')
            profile(ts)
        return ts
    def remove_classes(ts, labels=[], other_label='Control'):
        exclude_set = labels if len(labels)>0 else [other_label, ]
        col_target = TSet.target_field
        N0 = ts.shape[0]
        ts = ts.loc[~ts[col_target].isin(exclude_set)]    # isin([other_label, ])
        print('... remove labels: %s > size(ts): %d -> %d' % (labels, N0, ts.shape[0]))         
        return ts
    def subsample(ts, n=None, random_state=53):
        # maxNPerClass = kargs.get('n_per_class', None)
        n_per_class = n
        if n_per_class: # 0 or None => noop
            ts = sampling.sampleDataframe(ts, col=TSet.target_field, n_per_class=n_per_class, random_state=random_state)
            n_classes_prime = check_tset(ts)
            print('... after subsampling, size(ts)=%d, n_classes=%d (same?)' % (ts.shape[0], n_classes_prime))
        else: 
            # noop 
            pass 
        return ts

    # import vector
    from tset import TSet  
    from sampler import sampling
    # from sklearn.preprocessing import StandardScaler, MaxAbsScaler # feature scaling for sparse repr. 

    # w2v_method = kargs.get('w2v_method', vector.D2V.w2v_method) 
    
    # parameters for tset identifier
    # config()
    # tsHandler.dir_type = 'combined'
    # sequence content: mixed, diag only? med only? 
    # [note] training data by default are derived from the coding sequences consisting of diag and med codes
    # seq_ptype = 'regular' # seqparams.normalize_ctype(**kargs) # sequence pattern type: regular, random, diag, med, lab

    # [params] loading training data
    #     index: used to distinguish among multiple training and test sets
    #     suffix: served as secondary ID (e.g. training data made from sequences of different contents)
    #     meta: user-defined file ID (default='D')
    ts = tsHandler.load(index=kargs.get('index', 0), meta=kargs.get('meta', None)) 
    
    ### scaling is unnecessary for d2v
    if kargs.get('scale_', False): 
        pass
    #     # scaler = StandardScaler(with_mean=False)
    #     scaler = MaxAbsScaler()
    #     X = scaler.fit_transform(X)

    ### modify classes
    ts = customize_tset(ts)  # <- label_map, focused_labels
    n_classes0 = check_tset(ts)

    # drop explicit control data in multiclass
    if kargs.get('drop_ctrl', False): # drop control group (e.g. label: 'Control')
        assert n_classes0 > 2, "Not a multiclass domain (n_classes=%d)" % n_classes0
        ts = remove_classes(ts, labels=['Control', ])

    maxNPerClass = kargs.get('n_per_class', None) 
    if maxNPerClass is not None: 
        assert maxNPerClass > 1
        # prior to samplig, need to convert to canonical labels first 
        ts = subsample(ts, n=maxNPerClass)

    return ts # dataframe or None (if no throw)

def loadTSetChunk(**kargs):
    def profile(ts): 
        return tsHandler.profile(ts) 
    def customize_tset(ts): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Control"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            ts = focus(ts, labels=focused_classes, other_label='Control')
        lmap = kargs.get('label_map', seqparams.System.label_map)
        if lmap: # relabel so that classes only retain the main CKD stages
            print('(customize_tset) Modifying training data ...\n> Prior to re-labeling ...')
            profile(ts)
            ts = merge(ts, lmap)
            print('... After re-labeling ...')
            profile(ts)
        return ts
    def remove_classes(ts, labels=[], other_label='Control'):
        exclude_set = labels if len(labels)>0 else [other_label, ]
        col_target = TSet.target_field
        N0 = ts.shape[0]
        ts = ts.loc[~ts[col_target].isin(exclude_set)]    # isin([other_label, ])
        print('... remove labels: %s > size(ts): %d -> %d' % (labels, N0, ts.shape[0]))         
        return ts
    def check_tset(ts): 
        target_field = TSet.target_field
        n_classes = 1 
        no_throw = kargs.get('no_throw', True)
        if ts is not None and not ts.empty: 
            # target_field in ts.columns # o.w. assuming that only 1 class 
            n_classes = len(ts[target_field].unique())  
            print('loadTSet> number of classes: %d' % n_classes)
        else:
            msg = 'loadTSet> Warning: No data found (cohort=%s)' % kargs.get('cohort', 'CKD')
            if no_throw: 
                print msg 
            else: 
                raise ValueError, msg
        return n_classes
    from tset import TSet

    chunksize = kargs.get('chunksize', 1000)
    for ts in tsHandler.load_chunk(index=kargs.get('index', 0), meta=kargs.get('meta', None), chunksize=chunksize): 
        
        ### modify classes
        ts = customize_tset(ts)  # <- label_map, focused_labels
        n_classes0 = check_tset(ts)

        # drop explicit control data in multiclass
        if kargs.get('drop_ctrl', False): # drop control group (e.g. label: 'Control')
            assert n_classes0 > 2, "Not a multiclass domain (n_classes=%d)" % n_classes0
            ts = remove_classes(ts, labels=['Control', ])

        yield ts

def loadTSetChunk2(fpath, **kargs):
    """
    Similar to loadTSetchunk() but assume that the input path is given. 
    This is essentially a wrapper of panda's read_csv(). 

    Use
    ---
    1. Use tsHandler.load_from() to obtained the path to the training data, where path is configured 
       via parameters provided to tsHandler including cohort, sequence type (ctype), among others. 

    """
    def profile(ts): 
        return tsHandler.profile(ts) 
    def customize_tset(ts): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Control"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            ts = focus(ts, labels=focused_classes, other_label='Control')
        lmap = kargs.get('label_map', seqparams.System.label_map)
        if lmap: # relabel so that classes only retain the main CKD stages
            print('(customize_tset) Modifying training data ...\n> Prior to re-labeling ...')
            profile(ts)
            ts = merge(ts, lmap)
            print('... After re-labeling ...')
            profile(ts)
        return ts
    def remove_classes(ts, labels=[], other_label='Control'):
        exclude_set = labels if len(labels)>0 else [other_label, ]
        col_target = TSet.target_field
        N0 = ts.shape[0]
        ts = ts.loc[~ts[col_target].isin(exclude_set)]    # isin([other_label, ])
        print('... remove labels: %s > size(ts): %d -> %d' % (labels, N0, ts.shape[0]))         
        return ts
    def check_tset(ts): 
        target_field = TSet.target_field
        n_classes = 1 
        no_throw = kargs.get('no_throw', True)
        if ts is not None and not ts.empty: 
            # target_field in ts.columns # o.w. assuming that only 1 class 
            n_classes = len(ts[target_field].unique())  
            print('loadTSet> number of classes: %d' % n_classes)
        else:
            msg = 'loadTSet> Warning: No data found (cohort=%s)' % kargs.get('cohort', 'CKD')
            if no_throw: 
                print msg 
            else: 
                raise ValueError, msg
        return n_classes

    import pandas as pd 

    chunksize = kargs.get('chunksize', 1000)
    for i, ts in enumerate(pd.read_csv(fpath, chunksize=chunksize, iterator=True)):
        ### modify classes
        ts = customize_tset(ts)  # <- label_map, focused_labels
        n_classes0 = check_tset(ts)

        # drop explicit control data in multiclass
        if kargs.get('drop_ctrl', False): # drop control group (e.g. label: 'Control')
            assert n_classes0 > 2, "Not a multiclass domain (n_classes=%d)" % n_classes0
            ts = remove_classes(ts, labels=['Control', ])
        
        yield ts

def loadSparseTSet(**kargs): 
    def profile(X, y): 
        n_classes = len(np.unique(y))
        no_throw = kargs.get('no_throw', True)

        print('loadSparseTSet> X (dim: %s), y (dim: %d)' % (str(X.shape), len(y)))
        print("                + number of store values (X): %d" % X.nnz)
        print('                + number of classes: %d' % n_classes)

        # assert X.shape[0] == len(y)
        return    
    def get_tset_id(ts=None): # L <-
        return tsHandler.get_tset_id(ts)  
    def customize_tset(X, y): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Control"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            y = focusLabels(y, labels=focused_classes, other_label='Control')  # output in np.ndarray

        lmap = kargs.get('label_map', seqparams.System.label_map)
        if lmap: 
            profile(X, y)
            y = mergeLabels(y, lmap=lmap)
            print('> After re-labeling ...')
            profile(X, y)
        return (X, y)
    def remove_classes(X, y, labels=[], other_label='Control'):
        exclude_set = labels if len(labels)>0 else [other_label, ]
        
        N0 = len(y)
        ys = Series(y)
        cond_pos = ~ys.isin(exclude_set)

        idx = ys.loc[cond_pos].index.values
        y = ys.loc[cond_pos].values 
        X = X[idx]

        print('... remove labels: %s > size(ts): %d -> %d' % (labels, N0, X.shape[0]))         
        return (X, y)
    def get_tset_dir(): 
        return tsHandler.get_tset_dir()
    def subsample(X, y, n=None, sort_index=False, random_state=53):
        """
        Subsampling operation applied after doc vec X is derived. 
        """
        if n is not None: 
            idx = cutils.samplePerClass(y, n_per_class=n, sort_index=sort_index, random_state=random_state)
            print('  + X (dim0: %s -> dim: %s)' % (X.shape[0], len(idx)))
            return (X[idx], y[idx])
        return (X, y)

    from sklearn.preprocessing import StandardScaler, MaxAbsScaler # feature scaling for sparse repr.
    import classifier.utils as cutils

    # [note] use TSet.loadSparse0(...) to include docIDs with a return value (X, y, docIDs)
    #        example: tset-IDregular-bow-regular-GCKD.npz
    X, y = tsHandler.loadSparse(index=kargs.get('index', None), meta=kargs.get('meta', None)) 
    
    ### scaling X
    if kargs.get('scale_', False): 
        # scaler = StandardScaler(with_mean=False)
        scaler = MaxAbsScaler()
        X = scaler.fit_transform(X)

    ### modify classes
    if X is not None and y is not None: 
        X, y = customize_tset(X, y)

    # drop explicit control data in multiclass
    n_classes0 = len(np.unique(y)) 
    if kargs.get('drop_ctrl', False): # drop control group (e.g. label: 'Control')
        assert n_classes0 > 2, "Not a multiclass domain (n_classes=%d)" % n_classes0
        X, y = remove_classes(X, y, labels=['Control', ])

    ### subsampling 
    maxNPerClass = kargs.get('n_per_class', None)
    if maxNPerClass:
        assert maxNPerClass > 1
        y = np.array(y)
        X, y = subsample(X, y, n=maxNPerClass)

    return (X, y)


def mergeLabels(L, lmap={}):
    # from seqparams import System
    import numpy as np

    if not lmap: lmap = seqparams.System.label_map

    Ls = Series(L)
    labelSet = Ls.unique()
    print('  + (before) unique labels:\n%s\n' % labelSet)
    for label, eq_labels in lmap.items(): 
        print('  + %s <- %s' % (label, eq_labels))
        cond_pos = Ls.isin(eq_labels)
        Ls.loc[cond_pos] = label  # rename to designated label 

    labelSet2 = Ls.unique()   
    print('mergeLabels> n_labels: %d -> %d' % (len(labelSet), len(labelSet2)))
    print('  + (after) unique labels:\n%s\n' % labelSet2)

    return np.array(Ls)
def merge(ts, lmap={}): 
    """
    
    lmap: label map: representative label -> equivalent labels

    Example 
    -------
    1. CKD cohort 
       Given 11 labels: 

       [‘G1-control’, ‘G1A1-control’,  ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ‘Stage 3b’, 
     ’Stage 4’, ‘Stage 5’, ‘ESRD after transplant’, ‘ESRD on dialysis’, ‘Unknown’]

       merge the 2 ESRD-related classes to 'Stage 5'

       then specify

       lmap['Stage 5'] = ['Stage 5', 'ESRD after transplant', ‘ESRD on dialysis’] 

       Note that 'Stage 5' itself needs not be in the list


       Example map
       lmap['Control'] = [‘G1-control’, ‘G1A1-control’, 'Unknown']  ... control data 
       lmap['Stage 5'] = [‘ESRD after transplant’, ‘ESRD on dialysis’]
       
       classes that map to themselves do not need to be specified

    Related
    -------
    binarize() 
    focus()
    """
    from tset import TSet  # [todo] use seqmaker.tset

    if not lmap: return ts

    # N = ts.shape[0]
    col_target = TSet.target_field
    labelSet = ts[col_target].unique()
    print('  + (before) unique labels:\n%s\n' % labelSet)
    for label, eq_labels in lmap.items(): 
        print('  + %s <- %s' % (label, eq_labels))
        cond_pos = ts[col_target].isin(eq_labels)
        ts.loc[cond_pos, col_target] = label  # rename to designated label 

    labelSet2 = ts[col_target].unique()   
    print('merge> n_labels: %d -> %d' % (len(labelSet), len(labelSet2)))
    print('  + (after) unique labels:\n%s\n' % labelSet2)

    return ts

def focusLabels(L, labels, other_label='Control'):
    """
    Out of all the possible labels in L, focus only on the labels in 'labels'

    """
    if not hasattr(labels, '__iter__'): 
        labels = [labels, ]

    N = len(L)
    Ls = Series(L)
    all_labels = Ls.unique()
    assert len(set(labels)-set(all_labels))==0, "Some of the focused classes %s are not part of the label set:\n%s\n" % (labels, str(labels))

    cond_pos = Ls.isin(labels)
    lsp = Ls.loc[cond_pos]
    lsn = Ls.loc[~cond_pos] # other classes
    
    print('focus> n_focus: %d, n_other: %d' % (lsp.shape[0], lsn.shape[0]))
    lsn.loc[:] = other_label

    return pd.concat([lsp, lsn]).sort_index().values # recover original indexing; ignore_index <- False by default
def focus(ts, labels, other_label='Control'): 
    """
    Consolidate multiclass labels into smaller categories. 
    Focuse only on the target labels and classify all of the other labels
    using as others.  

    Example: CKD has 11 classes, some of which are not stages. 
             Want to only focus on stage-related labels (Stage 1 - Stage 5), and 
             leave all the other labels, including unknown, to the other category

    Related
    -------
    binarize() 
    merge()


    Memo
    ----
    1. CKD dataset 
       [‘G1-control’, ‘G1A1-control’,  ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ‘Stage 3b’, 
     ’Stage 4’, ‘Stage 5’, ‘ESRD after transplant’, ‘ESRD on dialysis’, ‘Unknown’]
    
    """
    from tset import TSet  # [todo] use seqmaker.tset

    if not hasattr(labels, '__iter__'): 
        labels = [labels, ]

    N = ts.shape[0]
    col_target = TSet.target_field
    all_labels = ts[col_target].unique()
    assert len(set(labels)-set(all_labels))==0, "Some of the focused classes %s are not part of the label set:\n%s\n" % (labels, str(labels))

    cond_pos = ts[col_target].isin(labels)
    tsp = ts.loc[cond_pos]
    tsn = ts.loc[~cond_pos] # other classes
    
    print('focus> n_focus: %d, n_other: %d' % (tsp.shape[0], tsn.shape[0]))
    tsn.loc[:, col_target] = other_label

    return pd.concat([tsp, tsn]).sort_index()  # recover original indexing



