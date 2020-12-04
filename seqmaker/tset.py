# encoding: utf-8

import numpy as np
import multiprocessing

from gensim.models import Doc2Vec
import gensim.models.doc2vec

import seqparams
import os, sys, re, random

import scipy
from pandas import DataFrame, Series
import pandas as pd

# from batchpheno import dfUtils

p_tset = re.compile(r'(?P<prefix>tset)\-(?P<id>[-\w]+)\.(?P<ext>\w+)')

class TSet(seqparams.TSet):
    """

    Methods
    -------
    getName(): name the training data file

    prepareTrainingSet(X, ): 
        given the matrix (X) consisting of document vectors, turn X into 
        a dataframe and save it according to the naming convention enforced by 
        getName()

    getHeader(): 
        standardize the columns of the training set dataframe

    """
    @staticmethod
    def prepareTSetMultiLabel(X, **kargs):  # [X | y1 | y2 ... | yn]
        pass 

    @staticmethod
    def getFileId(fname):
        m = p_tset.match(fname)
        if m: 
            return m.group('id') 
        return None

    @staticmethod
    def canonicalize(ts, meta_fields=None):
        # from batchpheno import dfUtils
        if meta_fields is None: meta_fields = TSet.meta_fields  # [target_field, target_type, content_field, label_field, index_field, date_field, annotated_field, ]
        return drop(ts, fields=meta_fields)
    @staticmethod
    def toXY(ts, meta_fields=None, target_field=None):
        """
        Extract (X, y) from training set dataframe 
        where X: 
              y: 
        """ 
        if target_field is None: target_field = TSet.target_field

        y = [1] * ts.shape[0]
        if target_field in ts.columns:
            y = ts[target_field].values  # [t for t in ts[TSet.target_field]]
        else: 
            print('toXY> Warning: No target_field found (unsupervised learning?)')
        
        tsc = TSet.canonicalize(ts, meta_fields=meta_fields)
        X = tsc.values
        # print("transform> dim of X: %s, y: %s" % (str(X.shape), str(y.shape)))
        return (X, y)
    @staticmethod
    def toXY2(ts, n_features, target_field=None):
        if target_field is None: target_field = TSet.target_field

        y = [1] * ts.shape[0]
        if target_field in ts.columns:
            y = ts[target_field].values  # [t for t in ts[TSet.target_field]]
        else: 
            print('toXY2> Warning: No target_field found (unsupervised learning?)')
       
        X = ts[TSet.getFeatureColumns(n_features)].values

        return (X, y)

    @staticmethod
    def save(X, **kargs):
        kargs['save_']=True
        return TSet.prepareTrainingSet(X, **kargs)  
    @staticmethod
    def saveDoc(D, T=[], L=[], **kargs): 
        """
        Save the (transformed) MDS, parallel to the training set, and its meta data (T, L).  
        """
        raise ValueError, "Use tdoc.TDoc.save()"

    @staticmethod
    def loadSparseMatrix(cohort, **kargs): 
        import pandas as pd
        import scipy.sparse

        # TSet coordinate (cohort, seq_ptype, d2v_method)
        d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method)  # more specifically, use vector.D2V.d2v_method
        seq_ptype = kargs.get('seq_ptype', 'regular')
        suffix = kargs.get('suffix', None) # secondary ID, e.g. LCS training set

        # set index to None to exclude it
        inputdir_default = TSet.getPath(cohort=cohort, dir_type=kargs.get('dir_type', 'combined'))
        rootdir = kargs.get('inputdir', inputdir_default)
        assert os.path.exists(rootdir), "Invalid input directory:\n%s\n" % rootdir

        fname = kargs.get('fname', None)  # allow for user input
        if fname is None: 
            fname = TSet.getName(cohort=cohort, d2v_method=d2v_method, index=kargs.get('index', None), 
                                        seq_ptype=seq_ptype, suffix=suffix, ext='npz') # [params] suffix (secondary ID, e.g. LCS training set)
        
        fpath = fpathX = os.path.join(rootdir, fname)
        print('TSet.loadSparse> loading training set (cohort=%s, suffix=%s) from:\n%s\n' % (cohort, suffix, fpath))

        ### load X
        X = None # np.array([])
        nrow = 0
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
            X = scipy.sparse.load_npz(fpath)
            nrow = X.shape[0]
            print('TSet.loadSparse> X (dim=%s)' % str(X.shape))
        else: 
            msg = '  + Warning: training set (cohort=%s) does not exist in:\n%s\n' % (cohort, fpath)
            # raise ValueError, msg
            print msg

        ### load y
        y = docIDs = None
        fstem, fext = os.path.splitext(fname)
        
        fnameY = '%s.csv' % fstem
        fpathY = os.path.join(rootdir, fnameY)
        if os.path.exists(fpathY) and os.path.getsize(fpathY) > 0: 
            ts = pd.read_csv(fpathY, sep=',', header=0, index_col=False, error_bad_lines=True)
                    
            y = ts[TSet.target_field].values  # all positive
            docIDs = ts[TSet.index_field] 

            n_classes = len(np.unique(y))
            assert len(y) == X.shape[0], "Inconsistent dim: X(%d) <> y(%d)" % (X.shape[0], y.shape[0])
            print('TSet.loadSparse> y (dim=%d), n_classes=%d' % (y.shape[0], n_classes))
        else: 
            msg = '  + labels do not exist.'
            print msg

        # load fset
        fset = []
        
        fname_fset = '%s-fset.csv' % fstem
        fpath_fset = os.path.join(rootdir, fname_fset)
        if os.path.exists(fpath_fset) and os.path.getsize(fpath_fset) > 0: 
            ts = pd.read_csv(fpath_fset, sep='|', header=0, index_col=False, error_bad_lines=True)    
            fset = ts['lcs'].values    
            
            assert len(fset) == X.shape[1], "Incompatible dimensions: nF=%d <> ncol(X)=%d" % (len(fset), X.shape[1])
            print('TSet.loadSparse> size(fset): %d' % len(fset))

        return (X, y, fset)  # or (X, y, docIDs)? 

    @staticmethod
    def loadSparse(cohort, **kargs):
        X, y, fset = TSet.loadSparseMatrix(cohort, **kargs)
        return (X, y, fset) 

    @staticmethod
    def saveSparse(X, **kargs):  # paired with loadSparse()
        return TSet.saveSparseMatrix(X, **kargs) 
    @staticmethod
    def saveSparseMatrix(X, **kargs): 
        """
        Save the sparse matrix (X), labels (y) if available, and 
        feature set (header)

        Params
        ------
        file ID: cohort, d2v_method, seq_ptype, index, suffix
        directory: cohort, dir_type


        Byproduct
        ---------
        1. .npz file for input sparse matrix X 
            via 
                scipy.sparse.save_npz(fpath, X)   

        2. .csv file for labels y (optional)
           via 
        
                ts.to_csv(fpath_y, sep=',', index=False, header=True)  .csv

        """
        import scipy.sparse
        # from pandas import DataFrame

        cohort_name = kargs.get('cohort', None)
        dir_type = kargs.get('dir_type', 'combined')
        outputdir = TSet.getPath(cohort=cohort_name, dir_type=dir_type) # see seqparams.TSet.getPath
        if kargs.has_key('outputdir'): outputdir = kargs['outputdir']
        # assert os.path.exists(outputdir), "Invalid outputdir (cohort=%s, dir_type=%s):\n%s\n" % (cohort_name, dir_type, outputdir)

        y = kargs.get('y', None)
        fset = kargs.get('header', [])  # feature set
        if kargs.get('save_', True): 
            assert os.path.exists(outputdir), "Invalid output path: %s" % outputdir

            n_classes = 1
            if y is not None: 
                # labelSet = np.unique(y)
                n_classes = len(np.unique(y))

            fname = kargs.get('outputfile', None)
            if fname is None: 
                d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method)
                # cohort_name = kargs.get('cohort', 'generic')
                seq_ptype = kargs.get('seq_ptype', 'regular')

                # user-defined file ID
                suffix = kargs.get('suffix', None) # additional info 
                if suffix is None: suffix = kargs.get('meta', None)

                # other meta data 

                # index: used to distinguish among multiple training and test sets
                # suffix: served as secondary ID (e.g. training data made from sequences of different contents, tset with labeling='lcs')
                fname = fname_X = TSet.getName(cohort=cohort_name, d2v_method=d2v_method, index=kargs.get('index', None), 
                                        seq_ptype=seq_ptype, suffix=suffix, ext='npz')
                print('TSet.saveSparse> tset params (cohort:%s d2v:%s, ctype:%s, suffix=%s) > fnames: X: %s' % \
                        (cohort_name, d2v_method, seq_ptype, suffix, fname))
                
            ### save X
            fpath = fpath_X = os.path.join(outputdir, fname)
            print('TSet.saveSparse> Saving (n_classes=%d, d2v method=%s, suffix=%s) training data X (dim: %s) to:\n%s\n' % \
                (n_classes, kargs.get('d2v_method', '?'), kargs.get('suffix', None), str(X.shape), fpath))

            # scipy.sparse.coo_matrix   ... coordinate format
            # np.savez(fpath, data=X.data, row=X.row, col=X.col, shape=X.shape)
            scipy.sparse.save_npz(fpath, X)

            ### save y: labels saved separately?
            fstem, fext = os.path.splitext(fname)
            assert fext.find('csv') == -1

            fname_y = '%s.csv' % fstem
            if y is not None: 
                assert len(y) == X.shape[0], "Incompatible dimensions: y(%d) <> X(%d)" % (len(y), X.shape[0])
                ts = DataFrame()
                ts[TSet.target_field] = y   # all positive

                idx = kargs.get('docIDs', [])
                if not idx: idx = range(X.shape[0])
                ts[TSet.index_field] = idx  # keep track of the original document ordering and associations with document segments

                fpath_y = os.path.join(outputdir, fname_y)
                print('saveSparse> Saving labels to .csv: %s' % fpath_y)
                ts.to_csv(fpath_y, sep=',', index=False, header=True)
            
            ### save feature set (fset)
            fname_fset = '%s-fset.csv' % fstem
            if len(fset) > 0: 
                assert len(fset) == X.shape[1], "Incompatible dimensions: nF=%d <> ncol(X)=%d" % (len(fset), X.shape[1])
                ts = DataFrame(fset, columns=['lcs', ])
                
                fpath_fset = os.path.join(outputdir, fname_fset)
                print('saveSparse> Saving feature set to .csv: %s' % fpath_fset)
                ts.to_csv(fpath_fset, sep='|', index=False, header=True)

        else: 
            print('saveSparseMatrix> save_ set to False, noop ...')

        return

    @staticmethod
    def toCSV(X, **kargs): 
        return TSet.prepareTrainingSet(X, **kargs)
    @staticmethod
    def prepareTrainingSet(X, **kargs):
        """
        Prepare training set data for single-label classificaitons. 

        Input: (X, y)

        Output: training data in dataframe format (ready to save)

        ts: training data in 2D array 

        kargs
        -----
            y: labels

            To save, need more params 

            shuffle_: shuffle training instances 
                tip: this should be set to False for visit-segemtned documents in order to preserve the order of timestamps 

            save_: set to True
            outputdir 
            outputfile (optional)

            d2v_method 
            cohort

            index: a number to distinguish variations of train-test split due to sampling or CV

            dir_type: e.g. None, 'train', 'test'

                if None: 
                    ./data/<cohort>/

                if 'train' (or any valid string such as 'test')
                    ./data/<cohort>/train


        """
        from pandas import DataFrame
        import random, os

        X = np.array(X) # if list of lists => 2D-array
        # X = ts.values
        nDoc = X.shape[0]
        idx = kargs.get('docIDs', [])
        if len(idx) == 0: idx = range(nDoc)

        header = kargs.get('header', None)
        if header is None: header = TSet.getHeader(X)  # automatically generated header
        ts = DataFrame(X, columns=header)

        labels = kargs.get('y', np.ones(X.shape[0]))  # or 'labels'
       
        # condition: signle-label format 
        assert not hasattr(labels[random.randint(0, len(labels)-1)], '__iter__'), "Input labels in multilabel format:\n%s\n" % labels[:10]

        n_classes = len(set(labels)); print('prepareTrainingSet> n_classes: %d' % n_classes)
        ts[TSet.target_field] = labels   # all positive

        # keep track of the original document ordering and associations with document segments
        # [note] this is especially important for i) sampled documents ii) visit-segmented documents
        ts[TSet.index_field] = idx  

        if kargs.get('shuffle_', False): 
            ts = ts.reindex(np.random.permutation(ts.index)) 

        kargs['delimit'] = ','

        if kargs.get('save_', False): 
            ts = TSet.saveDataFrame(ts, **kargs)
        return ts

    @staticmethod
    def saveDataFrame(df, **kargs): 
        """
        Save dataframe such as LCS (as features) and their document frequencies. 

        Params
        ------
        cohort 
        dir_type

        d2v_method, seq_ptype, suffix

        """
        sep = kargs.get('delimit',  ',')  # may want to use '|' 

        # output directory may depend on cohort, (which if not given, a System.cohort will be used)
        # getPath will create a new directory by default if not already existed 
        outputdir = TSet.getPath(cohort=kargs.get('cohort', None), dir_type=kargs.get('dir_type', 'combined')) # see seqparams.TSet.getPath
        if kargs.has_key('outputdir'): 
            outputdir = kargs['outputdir']
        assert os.path.exists(outputdir), "Invalid training set output directory: %s" % outputdir

        # by default, training data is not saved (to allow for dynamically generated training set)
        if True: 
            assert os.path.exists(outputdir), "Invalid output path: %s" % outputdir

            fname = kargs.get('outputfile', None)
            if fname is None: 
                d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method)
                cohort_name = kargs.get('cohort', 'generic')
                seq_ptype = kargs.get('seq_ptype', 'regular')

                # user-defined file ID
                suffix = kargs.get('suffix', None) # additional info 
                if suffix is None: suffix = kargs.get('meta', None)
                # other meta data 

                # index: used to distinguish among multiple training and test sets
                # suffix: served as secondary ID (e.g. training data made from sequences of different contents, tset with labeling='lcs')
                fname = TSet.getName(cohort=cohort_name, d2v_method=d2v_method, index=kargs.get('index', None), 
                                        seq_ptype=seq_ptype, suffix=suffix)
                print('TSet.saveDataFrame> tset params (cohort:%s d2v:%s, ctype:%s, suffix=%s)' % \
                    (cohort_name, d2v_method, seq_ptype, suffix))
                
            fpath = os.path.join(outputdir, fname)
            tVerifyPath = kargs.get('verfiy_path', False)
            tDoSave = True

            # save only when the dataset doesn't exist
            if tVerifyPath and (os.path.exists(fpath) and os.path.getsize(fpath) > 0): 
                print('TSet.saveDataFrame> File exists (cohort=%s, d2v=%s, suffix=%s) at:\n%s\n' % \
                        (kargs.get('cohort', '?'), kargs.get('d2v_method', '?'), kargs.get('suffix', None), fpath))
                tDoSave = False  # why? because sometimes saving operation can be time consuming too for large dataset
                
            if tDoSave: 
                print('TSet.saveDataFrame> Saving (cohort=%s, d2v=%s, suffix=%s) dataset to:\n%s\n' % \
                    (kargs.get('cohort', '?'), kargs.get('d2v_method', '?'), kargs.get('suffix', None), fpath))

                # incremental or all at once? 
                # if sys.getsizeof(df) > 10e9: # if size of dataframe > 10G

                df.to_csv(fpath, sep=sep, index=False, header=True) # cmp: previously, we use ts.index to keep labels => index=True 
        return df

    # @staticmethod
    # def fromCSV(**kargs): 
    #     ts = TSet.loadDataFrame(**kargs) 
    #     assert TSet.index_field in ts.columns and TSet.target_field in ts.columns
    #     return ts

    @staticmethod
    def loadDataFrame(**kargs):
        """


        Use
        ---
        1. load training data 
        2. load LCS features and their document frequencies
        """

        # set index to None to exclude it
        # TSet coordinate (cohort, seq_ptype, d2v_method)

        cohort = kargs.get('cohort', 'generic')
        d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method)
        seq_ptype = kargs.get('seq_ptype', 'regular')
        suffix = kargs.get('suffix', None) # secondary ID, e.g. LCS training set
        delimit = kargs.get('delimit', TSet.sep)  # ',' by default but for LCS feature files, may need to use '|'

        inputdir_default = TSet.getPath(cohort=cohort, dir_type=kargs.get('dir_type', 'combined'))
        rootdir = kargs.get('inputdir', inputdir_default)

        fname = kargs.get('inputfile', None)  # allow for user input
        if fname is None: 
            fname = TSet.getName(cohort=cohort, d2v_method=d2v_method, index=kargs.get('index', None), 
                                    seq_ptype=seq_ptype, suffix=suffix)  # [params] suffix (secondary ID, e.g. LCS training set)
        
        fpath = os.path.join(rootdir, fname)
        print('TSet.loadDataFrame> loading dataset (cohort=%s, suffix=%s) from:\n%s\n' % (cohort, suffix, fpath))

        df = None
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
            df = pd.read_csv(fpath, sep=delimit, header=0, index_col=False, error_bad_lines=True)
        else: 
            msg = 'TSet.load> Warning: dataset (cohort=%s) does not exist in:\n%s\n' % (cohort, fpath)
            # raise ValueError, msg
            print msg 
        return df

    @staticmethod
    def getFullPath(cohort, **kargs): # get full path to the training data (including the file)
        # TSet coordinate (cohort, seq_ptype, d2v_method)
        #      secondary params: suffix, index
        # TSet directory params: cohort, dir_type

        d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method)
        seq_ptype = kargs.get('seq_ptype', 'regular')
        suffix = kargs.get('suffix', None) # secondary ID, e.g. LCS training set

        # set index to None to exclude it
        fname = kargs.get('fname', None)  # allow for user input
        if fname is None: 
            fname = TSet.getName(cohort=cohort, d2v_method=d2v_method, index=kargs.get('index', None), 
                                    seq_ptype=seq_ptype, suffix=suffix)  # [params] suffix (secondary ID, e.g. LCS training set)
        inputdir_default = TSet.getPath(cohort=cohort, dir_type=kargs.get('dir_type', 'train'))
        rootdir = kargs.get('inputdir', inputdir_default)
        
        fpath = os.path.join(rootdir, fname)
        return fpath

    @staticmethod
    def load(cohort, **kargs): 
        """
        
        Params
        ------
        a. file name
        cohort
        d2v_method 
        seq_ptype
        index

        fname: specify file name specifically bypassing automatic file naming 

        b. prefix directory 
        (cohort) given by a)
        dir_type

        """
        import pandas as pd

        # TSet coordinate (cohort, seq_ptype, d2v_method)
        d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method)
        seq_ptype = kargs.get('seq_ptype', 'regular')
        suffix = kargs.get('suffix', None) # secondary ID, e.g. LCS training set

        tSparse = kargs.get('sparse', False)
        if tSparse: 
            return TSet.loadSparse(cohort, **kargs)  # output: (X, y, fset)

        # set index to None to exclude it
        fname = kargs.get('fname', None)  # allow for user input
        if fname is None: 
            fname = TSet.getName(cohort=cohort, d2v_method=d2v_method, index=kargs.get('index', None), 
                                    seq_ptype=seq_ptype, suffix=suffix)  # [params] suffix (secondary ID, e.g. LCS training set)
        inputdir_default = TSet.getPath(cohort=cohort, dir_type=kargs.get('dir_type', 'train'))
        rootdir = kargs.get('inputdir', inputdir_default)
        
        fpath = os.path.join(rootdir, fname)
        print('TSet.load> loading training set (cohort=%s, suffix=%s) from:\n%s\n' % (cohort, suffix, fpath))

        ts = None
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
            ts = pd.read_csv(fpath, sep=TSet.sep, header=0, index_col=False, error_bad_lines=True)
        else: 
            msg = 'TSet.load> Warning: training set (cohort=%s) does not exist in:\n%s\n' % (cohort, fpath)
            # raise ValueError, msg
            print msg

        return ts
    
    @staticmethod
    def loadChunks(cohort, **kargs): 
        """
        Similar to load() but used to load big dataset (e.g. >= 10G) that may not fit into memory. 

        """
        import pandas as pd

        fpath = TSet.loadFrom(cohort, **kargs)
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0: 
            msg = 'TSet.loadChunks> Warning: training set (cohort=%s) does not exist in:\n%s\n' % (cohort, fpath)
            # raise ValueError, msg
            print msg
            yield DataFrame()

        # generator
        chunkSize = kargs.get('chunksize', 1024)
        # [note] use pd.read_csv to read the csv file in chunks of 1000 lines with chunksize=1000 option
        for df in pd.read_csv(fpath, chunksize=chunksize, iterator=True):
            yield df

    @staticmethod
    def loadFrom(cohort, **kargs):
        # TSet coordinate (cohort, seq_ptype, d2v_method)
        d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method)
        seq_ptype = kargs.get('seq_ptype', 'regular')
        suffix = kargs.get('suffix', None) # secondary ID, e.g. LCS training set

        tSparse = kargs.get('sparse', False)
        assert tSparse is False, "loadFrom() does not apply to sparse representation for which X and y are kept in separate files."

        # set index to None to exclude it
        fname = kargs.get('fname', None)  # allow for user input
        if fname is None: 
            fname = TSet.getName(cohort=cohort, d2v_method=d2v_method, index=kargs.get('index', None), 
                                    seq_ptype=seq_ptype, suffix=suffix)  # [params] suffix (secondary ID, e.g. LCS training set)
        inputdir_default = TSet.getPath(cohort=cohort, dir_type=kargs.get('dir_type', 'train'))
        rootdir = kargs.get('inputdir', inputdir_default)
        
        fpath = os.path.join(rootdir, fname)
        print('TSet.loadFrom> loading training set (cohort=%s, suffix=%s) from:\n%s\n' % (cohort, suffix, fpath))
        
        return fpath

    @staticmethod
    def loadTrainSplit(cohort, **kargs): # d2v_method='pv-dm2',
        kargs['dir_type'] = 'train'
        return TSet.load(cohort, **kargs)
    @staticmethod
    def loadTestSplit(cohort, **kargs): # d2v_method='pv-dm2',
        # if d2v_method is None: d2v_method = seqparams.D2V.d2v_method
        kargs['dir_type']='test'
        return TSet.load(cohort, **kargs)
    @staticmethod
    def loadCombinedTSet(cohort, **kargs):   # seq_ptype
        # if d2v_method is None: d2v_method = seqparams.D2V.d2v_method
        kargs['dir_type']='combined'
        return TSet.load(cohort, **kargs)

    @staticmethod
    def loadLCSLabeledTSet(cohort, **kargs):
        return TSet.loadLCSTSet(cohort, **kargs) 
    @staticmethod
    def loadLCSTSet(cohort, **kargs): 
        """
        Load the training set in which LCSs are the labels. 
        """
        # if d2v_method is None: d2v_method = seqparams.D2V.d2v_method
        if not kargs.has_key('dir_type'): kargs['dir_type']='combined'  #
        
        suffix = 'Llcs'
        user_suffix = kargs.get('suffix', '')
        if user_suffix: suffix = '%s-%s' % (suffix, user_suffix) 

        kargs['suffix'] = suffix  # L: label
        return TSet.load(cohort, **kargs)

    @staticmethod
    def saveRankedLCSFSet(df, **kargs): 
        """
        Save a dataframe consisting of two columns 
        ['lcs', 'score']  where score can be document frequency, etc.

        """
        suffix = kargs.get('suffix', '')
        sysFileID = 'Rlcs'
        if not suffix: 
            suffix = sysFileID  # ranked LCS
        else: 
            suffix = '%s-%s' % (sysFileID, suffix)
        kargs['suffix'] = suffix  # meta?
        kargs['delimit'] = '|'
        TSet.saveDataFrame(df, **kargs)
        return 
    @staticmethod
    def loadRankedLCSFSet(**kargs):
        suffix = kargs.get('suffix', '')
        sysFileID = 'Rlcs'
        if not suffix: 
            suffix = sysFileID  # ranked LCS
        else: 
            suffix = '%s-%s' % (sysFileID, suffix)
        kargs['suffix'] = suffix  # meta?
        kargs['delimit'] = '|'
        return TSet.loadDataFrame(**kargs)

    @staticmethod
    def saveLCSFreqDistribution(df, **kargs):
        suffix = kargs.get('suffix', '')
        label = kargs.get('label', 'global')

        # remove spaces in label
        tok = '' 
        label = label.replace(' ', tok)

        isWeighted = kargs.get('is_weighted', False)

        sysFileID = 'WFreq-L' if isWeighted else 'Freq-L'
        if not suffix: 
            suffix = '%s%s' % (sysFileID, label)
        else: 
            suffix = '%s-%s%s' % (suffix, sysFileID, label)
        kargs['suffix'] = suffix
        kargs['delimit'] = '|'
        TSet.saveDataFrame(df, **kargs)
        return
    @staticmethod
    def loadLCSFreqDistribution(**kargs):
        suffix = kargs.get('suffix', '')
        label = kargs.get('label', 'global')

        # remove spaces in label
        tok = '' 
        label = label.replace(' ', tok)

        isWeighted = kargs.get('is_weighted', False)
        sysFileID = 'WFreq-L' if isWeighted else 'Freq-L'

        if not suffix: 
            suffix = '%s%s' % (sysFileID, label)  
        else: 
            suffix = '%s-%s%s' % (suffix, sysFileID, label)
        kargs['suffix'] = suffix  # meta?
        kargs['delimit'] = '|'
        return TSet.loadDataFrame(**kargs) 

    @staticmethod
    def loadLCSFeatureTSet(cohort, **kargs):
        """
        Load the training set in which LCSs are features. 

        """
        if not kargs.has_key('dir_type'): kargs['dir_type']='combined'  #

        suffix = 'Flcs'
        user_suffix = kargs.get('suffix', '')  # or use 'meta'
        
        if user_suffix: suffix = '%s-%s' % (suffix, user_suffix) 

        kargs['suffix'] = suffix  # F: feature 
        return TSet.load(cohort, **kargs)

    @staticmethod
    def loadSparseLCSFeatureTSet(cohort, **kargs):
        """
        Load the sparse training set in which LCSs are features. 

        Params
        ------
        file ID: cohort, d2v_method, seq_ptype, index, suffix
        directory: cohort, dir_type

        """
        if not kargs.has_key('dir_type'): kargs['dir_type']='combined'  #

        suffix = 'SFlcs' # sparse feature set
        user_suffix = kargs.get('suffix', '')  # or use 'meta'
        if user_suffix: suffix = '%s-%s' % (suffix, user_suffix) 

        kargs['suffix'] = suffix  # F: feature 

        # output: (X, y, fset)
        return TSet.loadSparse(cohort, **kargs)  # paired with loadSparse()
        
    @staticmethod
    def loadBaseTSet(cohort, **kargs): # tset prior to LCS-based labels => no secondary ID
        """


        Memo
        ----
        1. load existing training set
            params: cohort, d2v_method, seq_ptype, index, suffix/meta 
            directory: cohort, dir_type

        ts = TSet.load(cohort=tsHandler.cohort, 
                       d2v_method=tsHandler.d2v, 
                       seq_ptype=tsHandler.ctype, 
                       index=index,  # CV index (if no CV, then 0 by default)
                       dir_type=tsHandler.dir_type, 
                       suffix=tsHandler.meta if not meta else meta)
        """

        # if kargs.get('verbose', False): print('loadBaseTSet> ') 
        if not kargs.has_key('dir_type'): kargs['dir_type']='combined'  
        suffix = kargs.get('suffix', None) # or use 'meta'

        ts = TSet.load(cohort, **kargs) # file params: cohort, d2v_method, seq_ptype, index, suffix/meta 
        if ts is None or ts.empty: 
            if suffix is not None: 
                print('loadBaseTSet> Warning: secondary file ID is included: %s' % suffix)
                print('  + Try removing suffix and load the data again ...')
                kargs['suffix']=None 
                ts = TSet.load(cohort, **kargs)
                assert ts is not None and not ts.empty, "failed to load the plain d2v training set."
        return ts
    @staticmethod
    def loadNGramTSet(cohort, **kargs):
        if not kargs.has_key('dir_type'): kargs['dir_type']='combined'  # 
        kargs['suffix'] = 'Lngram'
        return TSet.load(cohort, **kargs)

    @staticmethod
    def saveLCSLabeledTSet(X, y, **kargs): 
        return TSet.saveLCSTSet(X, y, **kargs)
    @staticmethod
    def saveLCSTSet(X, y, **kargs):
        if not kargs.has_key('dir_type'): kargs['dir_type']='combined'  # 
        suffix = 'Llcs'
        user_suffix = kargs.get('suffix', '')
        if user_suffix: suffix = '%s-%s' % (suffix, user_suffix)
        kargs['suffix'] = suffix
        return TSet.save(X, y=y, **kargs)

    @staticmethod
    def saveLCSFeatureTSet(X, y, **kargs):
        if not kargs.has_key('dir_type'): kargs['dir_type']='combined'  # 
        header = kargs.get('header', None)
        if header is None: 
            # raise ValueError, "Feature names are not given" 
            print('saveLCSFeatureTSet> Warning: Feature names are not given.')
        else: 
            assert X.shape[1] == len(header), "size(header): %d but X[1]: %d" % (len(header), X.shape[1])
        suffix = 'Flcs'
        user_suffix = kargs.get('suffix', '')
        if user_suffix: suffix = '%s-%s' % (suffix, user_suffix)
        kargs['suffix'] = suffix
        return TSet.save(X, y=y, **kargs)
    @staticmethod
    def saveSparseLCSFeatureTSet(X, **kargs):
        if not kargs.has_key('dir_type'): kargs['dir_type']='combined'  # 
        header = kargs.get('header', [])
        if not header: 
            # raise ValueError, "Feature names are not given" 
            print('saveLCSFeatureTSet> Warning: Feature names are not given (assuming that variables defined via loadRankedLCSFSet())')
            # PS: feature set is saved separately in a .csv file (see rank_)
            #     load variables via 
            #        TSet.loadRankedLCSFSet(cohort=cohort, d2v_method=lcsHandler.d2v_method, 
            #                 seq_ptype=lcsHandler.ctype, suffix=lcsHandler.meta)
        else: 
            assert X.shape[1] == len(header), "size(header): %d but X[1]: %d" % (len(header), X.shape[1])
        suffix = 'SFlcs'  # sparse feature set
        user_suffix = kargs.get('suffix', '')
        if user_suffix: suffix = '%s-%s' % (suffix, user_suffix)
        kargs['suffix'] = suffix
        return TSet.saveSparse(X, **kargs) # kargs has 'y', 'fset'

    @staticmethod
    def getXY(cohort, d2v_method='pv-dm2', **kargs): # [params] dir_type, index
        ts = TSet.load(cohort, d2v_method=d2v_method, **kargs) 
        X, y = TSet.toXY(ts)
        return (X, y)

### end class TSet  

def drop(ts, fields):  # ported from batchpheno.dfUtils
    assert hasattr(fields, '__iter__')
    for field in fields: 
        if field in ts.columns: 
            ts = ts.drop(field, axis=1)
    # ts.drop(fields, axis=1, inplace=True)
    return ts 

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
       lmap['Others'] = [‘G1-control’, ‘G1A1-control’, 'Unknown']  ... control data 
       lmap['Stage 5'] = [‘ESRD after transplant’, ‘ESRD on dialysis’]
       
       classes that map to themselves do not need to be specified

    Related
    -------
    binarize() 
    focus()
    """
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

def focus(ts, labels, other_label='Others'):  # refactored from seqClassify
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

def loadTSetCombined(**kargs):  # refactored from seqClassify
    """
    Load training data. Similar to loadTSet() but assume no separation between train and test splits. 

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
    def modify_tset(ts): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Others"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            ts = focus(ts, labels=focused_classes, other_label='Others')
        lmap = kargs.get('label_map', seqparams.System.label_map)
        if lmap: # relabel so that classes only retain the main CKD stages
            print('loadTSetCombined> Modifying training data ...\n> Prior to re-labeling ...')
            profile(ts)
            ts = merge(ts, lmap)
            print('> After re-labeling ...')
            profile(ts)
        return ts
    def remove_classes(ts, labels=[], other_label='Others'):
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
            print('  + after subsampling, size(ts)=%d, n_classes=%d (same?)' % (ts.shape[0], n_classes_prime))
        else: 
            # noop 
            pass 
        return ts

    # import vector 
    from sampler import sampling
    from seqConfig import tsHandler  # configuration for seqmaker module (e.g. experimental settings, load/save training data)

    # from sklearn.preprocessing import StandardScaler, MaxAbsScaler # feature scaling for sparse repr. 

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
    ts = modify_tset(ts)  # <- label_map, focused_labels
    n_classes0 = check_tset(ts)

    # drop explicit control data in multiclass
    if kargs.get('drop_ctrl', False): # drop control group (e.g. label: 'Others')
        assert n_classes0 > 2, "Not a multiclass domain (n_classes=%d)" % n_classes0
        ts = remove_classes(ts, labels=['Others', ])

    maxNPerClass = kargs.get('n_per_class', None) 
    if maxNPerClass is not None: 
        assert maxNPerClass > 1
        # prior to samplig, need to convert to canonical labels first 
        ts = subsample(ts, n=maxNPerClass)

    return ts # dataframe or None (if no throw)

def mergeLabels(L, lmap={}):
    from seqparams import System

    if not lmap: lmap = System.label_map

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
       lmap['Others'] = [‘G1-control’, ‘G1A1-control’, 'Unknown']  ... control data 
       lmap['Stage 5'] = [‘ESRD after transplant’, ‘ESRD on dialysis’]
       
       classes that map to themselves do not need to be specified

    Related
    -------
    binarize() 
    focus()
    """
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

def focusLabels(L, labels, other_label='Others'):
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
def focus(ts, labels, other_label='Others'): 
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

def makeTSet(**kargs):  # refactored from seqCluster
    """
    Make training set data (based on w2v and d2v repr) for classification and clusterig
    
    Usage 
    -----
    1. try to call load_tset() first. 

    Related
    -------
    * t_preclassify*
    make_tset_labeled (n_classes >=2)

    Operations 
    --------- 
    loadModel: read (temp docs: D) -> analyze -> vectorize (: D -> w2v)

    Output
    ------ 
    1. identifier: 
         identifier = '%s-%s-%s' % (seq_ptype, w2v_method, d2v_method)
         'tset_nC%s_ID%s-G%s.csv' % (n_classes, identifier, cohort_name)

    """
    # import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from pattern import diabetes as diab 

    # [params] cohort   # [todo] use classes to configure parameters, less unwieldy
    composition = seq_compo = kargs.get('composition', 'condition_drug')
    cohort_name = kargs['cohort'] # kargs.get('cohort', 'diabetes')
    # doc_basename = seq_compo if seq_ptype.startswith('reg') else '%s-%s' % (seq_compo, seq_ptype) 
    # if cohort_name is not None: 
    #     doc_basename = '%s-%s' % (doc_basename, cohort_name)

    # [params]
    read_mode = seqparams.TDoc.read_mode  # assign 'doc' (instead of 'seq') to form per-patient sequences
    seq_ptype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular')) # sequence pattern type: regular, random, diag, med, lab

    # document embedding wrt all information
    assert seq_ptype in ['regular', ], "seq_ptype=%s > document vectors will not contain complete coding information." % seq_ptype

    testdir = seqparams.get_testdir() # prefix is None by default => os.getcwd(), topdir='test'

    # assumption: w2v has been completed
    # os.path.join(os.getcwd(), 'data/%s' % cohort_name) 
    basedir = outputdir = seqparams.getCohortLocalDir(cohort=cohort_name) # or sys_config.read('DataExpRoot')
    print('make_tset> basedir: %s' % basedir)
    
    # [params] D2V use seqparams and vector modules to configure (or simply use their defaults)
    # n_features = kargs.get('n_features', GNFeatures)
    # window = kargs.get('window', GWindow)
    # min_count = kargs.get('min_count', GMinCount)  # set to >=2, so that symbols that only occur in one doc don't count
    # n_cores = multiprocessing.cpu_count()
    # # print('info> number of cores: %d' % n_cores)
    # n_workers = kargs.get('n_workers', GNWorkers)

    # configure vector module 
    # vector.config(n_features=GNFeatures, window=GWindow, min_count=GMinCount)

    # doctype = 'timed'  # ['doc', 'timed', 'labeled']; see seqparams.TDoc

    # coding semantics lookup
    test_model = kargs.get('test_model', True) # independent from loading or (re)computing w2v
    bypass_code_lookup = kargs.get('bypass_lookup', True)  # this is time consuming involving qymed MED codes, etc. 

    load_labelsets = kargs.get('load_labelsets', False)

    # [input] sequences and word2vec model 
    # [note] seqAnalyzer expects a "processed" input sequences (i.e. .pkl or .csv) instead of the raw input (.dat)
    # ifiles = TDoc.getPaths(cohort=cohort_name, doctype=doctype, ifiles=kargs.get('ifiles', []), ext='csv', verfiy_=False) 
    # print('make_tset> loadModel> (cohort: %s => ifiles: %s)' % (cohort_name, ifiles)) 

    ### read + (analyze) + vectorize
    # [note] the 'analyze' step can be performed independently
    tSimplifyCode = kargs.get('simplify_code', False)

    # result = sa.loadModel(n_features=n_features, window=window, min_count=min_count, n_workers=n_workers, 
    #                         w2v_method=w2v_method, 
    #                         seq_ptype=seq_ptype, read_mode=read_mode, test_model=test_model, 
    #                             ifile=ifile, cohort=cohort_name, 
    #                             load_seq=False, load_model=load_word2vec_model, load_lookuptb=load_lookuptb, 
    #                             bypass_lookup=bypass_code_lookup) # attributes: sequences, lookup, model

    ### load model
    # 1. read | params: cohort, inputdir, doctype, labels, n_classes, simplify_code
    #         | assuming structured sequencing files (.csv) have been generated
    div(message='1. Read temporal doc files ...')

    # csv header: ['sequence', 'timestamp', 'label'], 'label' may be missing
    # [params] if 'complete' is set, will search the more complete .csv file first (labeled > timed > doc)
    
    # if result set (sequencing data is provided, then don't read and parse from scratch)
    ret = kargs.get('result_set', {})
    if not ret: 
        ret = readDocFromCSV(cohort=cohort_name, inputdir=basedir, ifiles=ifiles, complete=True) # [params] doctype (timed)
    seqx = result['sequence'] # must have sequence entry
    tseqx = result.get('timestamp', [])
    nDoc = n_docs = len(seqx)
    print('verify> number of docs: %d' % nDoc)

    ### data subset selection and sequence transformation 
    div(message='1.1 Data Transformation and subset selection ...')
    # [control] simplify code?
    if tSimplifyCode: seqx = seqAlgo.simplify(seqx)  # this will not affect medication code e.g. MED:12345
    ### [todo][control] train specific subset of codes (e.g. diagnostic codes only)
    # for i, doc in enumerate(seqx): 
    #     seqx[i] = st.transform(doc, policy=cut_policy, inclusive=True, 
    #                     seq_ptype=seq_ptype, cohort=cohort_name, predicate=condition_predicate)
    #     # time stamps 
    #     # labels

    ### determine labels and n_classes 
    div(message='1.2 Document labeling ...')
    # check user input -> check labels obtained from within he labeled_seq file
    labels = kargs.get('labels', result.get('labels', []))
    if not labels: 
        # disease module: pattern.medcode or pattern.<cohort> (e.g. diabetes cohort => pattern.diabetes)
        print("info> No data found from 1) user input 2) sequencing data > call getSurrogateLabels() from disease module.")  
        labels = getSurrogateLabels(seqx, cohort=cohort_name)  # arg: cohort_name
    labeling = True if len(labels) > 0 else False
    
    # [condition] len(seqx) == len(tseqx) == len(labels) if all available

    n_classes = 1  # seqparams.arg(['n_classes', ], default=1, **kargs) 
    if labeling: n_classes = len(set(labels))
    # if labeling and n_classes > 1: 
    #     print('status> labeling: %s, n_classes: %d > make training set assuming labels available ...' % (labeling, n_classes))
    #     kargs['labels'] = labels
    #     kargs['result_set'] = ret 
    #     return make_tset_labeled(**kargs)  # [I/O]
    # assert n_classes == 1, "We shall assume there exists only one class in an unlabeld training set! (n_classes=%d)" % n_classes
    # [condition] n_classes determined
    print('stats> n_docs: %d, n_classes: %d | cohort: %s, composition: %s' % (nDoc, n_classes, cohort_name, seq_compo))

    div(message='2. Compute document embedding (params: )') # [note] use W2V and D2V classes in seqparams and vector to adjust d2v parameters
    d2v_method = kargs.get('d2v_method', 'pv-dm')
    
    # model = vector.getDocVecModel(d2v_method=d2v_method)
    X = getDocVec(seqx, d2v_method=d2v_method, outputdir=outputdir, test_=test_model) # [params] w2v_method, outputdir, outputfile
    assert X.shape[0] == nDoc

    # lookuptb = result['symbol_chart']
    print('status> Model computation complete.')
    
    div(message='3. Save training set')
    # e.g. .../data/PTSD/tset_nC1_IDregular-sg-tfidfavg-GPTSD.csv
    fname = TSet.getName(cohort=cohort_name, d2v_method=d2v_method) # seq_ptype ('regular'), w2v_method ('sg')
    fpath = os.path.join(outputdir, fname) # nC: n_classes, G: group ie cohort
    ts = TSet.prepareTrainingSet(X, labels=labels) # [params] labels

    # fpath = os.path.join(basedir, 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc0, seq_ptype)) 
    print('io> saving (n_classes=%d, d2v_method=%s) training data to:\n%s\n' % (n_classes, d2v_method, fpath))
    ts.to_csv(fpath, sep=',', index=False, header=True) # cmp: previously, we use ts.index to keep labels => index=True

    return ts

def test(**kargs): 
    pass 
if __name__ == "__main__": 
    test()