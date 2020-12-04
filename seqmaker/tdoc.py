import numpy as np
import multiprocessing

import os, sys, re, random, time
from datetime import datetime 

# pandas 
from pandas import DataFrame
import pandas as pd 

# gensim 
from gensim.models import Doc2Vec
import gensim.models.doc2vec

# tpheno
import seqReader as sr
import seqparams, seqAlgo
import labeling 
from batchpheno.utils import div

gHasConfig = True
try: 
    from config import seq_maker_config, sys_config
except: 
    print("tdoc> Could not find default configuration package.")
    gHasConfig = False


#######################################################################################
#
#  Module for the Coding Sequence Document
#  
#
# 
#  Memo
#  ----
#  1. See seqparams for the base class definition
#
#######################################################################################


class TDoc(seqparams.TDoc): # this defines a different inheritance line from labeling.TDoc 

    # sequencing_subdir = 'sequencing'  # subdirectory specific for "sequencing data" 
    csv_sep = '|'

    # define these in base class at seqparams
    # header_timed_seq = ['sequence', 'timestamp', ]
    # header_labeled_seq = ['sequence', 'timestamp', 'label']

    filters = '!"#$%&()*+,/;<=>?@[\]^`{|}~'  # '.', ':', '_' '-' should be consiidered as part of the codes/tokens

    vocab_size = 20001 # default max vocabulary size 

    @staticmethod
    def to2DArray(df, col='sequence'): # [todo] consider visits
        """

        Memo
        ----
        1. seqparams.TDoc.strToSeq()

        """
        seqx = [] # a list of (lists of tokens)
        cSep, vSep, hSep = TDoc.token_sep, TDoc.token_end_visit, TDoc.token_end_history
        for seqstr in df[col].values: 
            if seqstr.find(hSep) >= 0: 
                assert seqstr[-1] == hSep, "Medical history separator must be at the end of the document."
                seqstr = seqstr[:-1]
            if seqstr.find(TDoc.token_end_visit) >= 0: 
                tokens = []
                visits = seqstr.split(TDoc.token_end_visit)  # ';'
                for visit in visits: 
                    tokens.extend(visit.split(sep))
            else: 
                tokens = seqstr.split(sep)
            seqx.append(tokens)
        return np.array(seqx)

    @staticmethod 
    def loadSrc(cohort, inputdir=None, ifiles=[], complete=True): 
        """

        Note
        ----
        1. This is not the same as load(), which load MCSs used to generate d2v training set. 

        """
        import seqReader as sr
        if inputdir is None: inputdir = sys_config.read('DataExpRoot')  # document source directory
        ret = sr.readDocFromCSV(cohort=cohort, inputdir=inputdir, ifiles=ifiles, complete=True) # [params] doctype (timed)
        # seqx = ret['sequence'] # must have sequence entry
        # tseqx = ret.get('timestamp', [])  

        return ret # keys: 'sequence', 'timestamp', 'label'

    @staticmethod
    def loadGeneric(filenames, prefix=None): 
        """
        load generic document sources (i.e. not cohort dependent) given file names. 

        Params
        ------
        prefix: the rootdir from which the given file set (filenames) are expected to be found. 
                if not given, sequencing dir by default. 
                e.g. <tpheno>/data-exp/sequencing

        """ 
        if prefix is None: 

            # e.g. .../tpheno/data-exp/sequencing
            prefix = os.path.join(sys_config.read('DataExpRoot'), TDoc.sequencing_subdir)
        ret.update(sr.readDocFromCSV(cohort='n/a', inputdir=prefix, ifiles=filenames, complete=True)) # [params] doctype (timed)
        return ret
 
    @staticmethod
    def getDY(cohort, inputdir=None, ifiles=[], complete=True, to_single_label=True):  # documents and labels
        ret = TDoc.loadSrc(cohort, inputdir=inputdir, ifiles=ifiles, complete=complete)

        seqx, tseqx, labelx = ret['sequence'], ret.get('timestamp', []), ret.get('label', [])
        nDoc = len(seqx); print('verify> number of docs: %d' % nDoc)  

        # class labels (NOT document labels)
        if not labelx: 
            # disease module: pattern.medcode or pattern.<cohort> (e.g. diabetes cohort => pattern.diabetes)
            print('getDY> No labels found in the documents of cohort: %s' % cohort)

            # use surrogate labels or just simply assume no labels (all positive)

            # labelx  = getSurrogateLabels(seqx, cohort=cohort)  # arg: cohort_name
            labelx = np.ones(nDoc)  # assume all positive

        # condition: if cohort is supported, then a list of numeric labels is returned; if not, then all positive
        print('test> original labels/tags: %s' % labelx[:10])
        # [condition] len(seqx) == len(tseqx) == len(labels) if all available

        # labels is not the same as tags (a list of lists)
        if to_single_label: 
            labelx = toSingleLabel(labelx, pos=0) # to single label format (for single label classification)
            print('loadDoc.test> labels (single label): %s' % labelx[:10]) 
        assert len(labelx) == len(seqx)

        n_classes = len(set(labelx))  # seqparams.arg(['n_classes', ], default=1, **kargs) 
        # [condition] n_classes determined
        print('stats> n_docs: %d, n_classes: %d | cohort: %s, composition: %s' % (nDoc, n_classes, cohort_name, seq_compo)) 

        return (seqx, labelx)

    @staticmethod
    def getTSetDocName(**kargs): 
        for param in ['cohort', ]: 
            assert kargs.has_key(param)
        return TDoc.getMCSName(**kargs)
    @staticmethod
    def getMCSName(**kargs):
        """
        Get the file name of the derived MCS file (transformed documents), which is created in parallel to 
        the training set created by document embedding. This file follows training set naming convention 
        instead of the source document convention as in {getName, getNameByContent}

        """
        from tset import TSet 

        ### parameters file naming
        d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method)
        cohort_name = kargs.get('cohort', 'generic')
        seq_ptype = kargs.get('seq_ptype', 'regular')
        suffix = kargs.get('suffix', None) # additional info 
        # other meta data 

        # index: used to distinguish among multiple training and test sets
        # suffix: served as secondary ID (e.g. training data made from sequences of different contents, tset with labeling='lcs')
        fname = TSet.getDocName(cohort=cohort_name, d2v_method=d2v_method, index=kargs.get('index', None), 
                                            seq_ptype=seq_ptype, suffix=kargs.get('suffix', None))  # file_stem <- 'mcs'

        return fname

    @staticmethod
    def relabel(df, col_target, **kargs): 
        lmap = kargs.get('label_map', {})
        if not lmap: return # noop 
        assert col_target in df.columns, "label attribute %s is not in the header:\n%s\n" % (col_target, df.columns.values) 
        labelSet = df[col_target].unique()
        print('  + (before) unique labels (n=%d):\n%s\n' % (len(labelSet), labelSet))
        for label, eq_labels in lmap.items(): 
            print('  + %s <- %s' % (label, eq_labels))
            cond_pos = df[col_target].isin(eq_labels)
            df.loc[cond_pos, col_target] = label  # rename to designated label 
        labelSet = df[col_target].unique()
        print('  + (after) unique labels (n=%d):\n%s\n' % (len(labelSet), labelSet))
        return # input dataframe (df) relabeled

    @staticmethod
    def parseRow(df, col='sequence', sep=','): # [note] assuming that visit separators ';' were removed
        seqx = [] # a list of (lists of tokens)
        
        isStr = False
        for i, row in enumerate(df[col].values):
            if isinstance(row, str): 
                isStr = True
            if i >= 3: break
            
        if isStr:  # sequence, timestamp
            for row in df[col].values:   # [note] label is not string
                tokens = row.split(sep)
                seqx.append(tokens)
        else: 
            # integer, etc. 
            seqx = list(df[col].values)

        return seqx

    @staticmethod
    def load(**kargs): 
        """
        Load the training documents (used to generate d2v training set.)
        
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
        import vector
        from tset import TSet # following the same naming convention except that 'tset' is replaced by 'mcs'

        # TSet coordinate (cohort, seq_ptype, d2v_method)
        d2v_method = kargs.get('d2v_method', vector.D2V.d2v_method)
        cohort_name = kargs.get('cohort', 'generic')
        seq_ptype = kargs.get('seq_ptype', 'regular')
        suffix = kargs.get('suffix', None) # secondary ID, e.g. LCS training set

        # set index to None to exclude it
        fname = kargs.get('inputfile', None)  # allow for user input
        if fname is None: 
            fname = TSet.getName(cohort=cohort_name, d2v_method=d2v_method, index=kargs.get('index', None), 
                                        seq_ptype=seq_ptype, suffix=kargs.get('suffix', None), 
                                        file_stem='mcs')  # file_stem <- 'mcs' instead of 'tset'
        inputdir_default = TSet.getPath(cohort=cohort_name, dir_type=kargs.get('dir_type', 'train')) # see seqparams.TSet.getPath
        rootdir = kargs.get('inputdir', inputdir_default)
        
        fpath = os.path.join(rootdir, fname)
        print('TSet.load> loading training set (cohort=%s, suffix=%s) from:\n%s\n' % (cohort, suffix, fpath))

        tsDoc = None
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
            tsDoc = pd.read_csv(fpath, sep=TDoc.csv_sep, header=0, index_col=False, error_bad_lines=True)  # TDoc.csv_sep='|'
        else: 
            msg = 'TSet.load> Warning: training document set (cohort=%s) does not exist in:\n%s\n' % (cohort, fpath)
            # raise ValueError, msg
            print msg

        ### parsing 
        ret = {} 
        header = kargs.get('header', ['sequence', 'timestamp', 'label', ])
        if not tsDoc.empty: 
            for h in header: 
                if h in tsDoc.columns: 
                    # or use seqparams.TDoc.strToSeq(df_seq, col='sequence')
                    # multilabel format e.g. '250.00,253.01' => ['250.00', '253.01']
                    print('TDoc.load> processing header %s ...' % h)
                    seqx = TDoc.parseRow(tsDoc, col=h, sep=',')
                    ret[h].extend(seqx)  # assuming that ';', '$' are all being replaced
                    # sdict[h] += len(seqx)

        return ret # key: ['sequence', 'timestamp', 'label', ] ~ (D, T, L)

    @staticmethod
    def save(D, T=[], L=[], **kargs): 
        """

        Note
        ----
        1. parallel to TSet.toCSV(), following the same file naming convention except that 
           file stem changes from 'tset' to 'mcs'

        """
        return TDoc.toCSV(D, T=T, L=L, **kargs)
    @staticmethod
    def toCSV(D, T=[], L=[], **kargs): 
        """
        Save the (transformed) MDS, parallel to the training set, and its meta data (T, L).  
        """
        import vector
        from tset import TSet  # this derived document must correspond to a training set

        ### params: file naming
        d2v_method = kargs.get('d2v_method', vector.D2V.d2v_method)
        cohort_name = kargs.get('cohort', 'generic')
        seq_ptype = kargs.get('seq_ptype', 'regular')

        # reserved for user-defined file ID
        suffix = kargs.get('suffix', None) # additional info 
        if suffix is None: suffix = kargs.get('meta', None)
        # other meta data 

        # [note] also see docProc.makeLabeledDocuments()
        #        indices: document IDs (useful when documents are segmented into constituent parts/paragraphs)
        df = docToCSV(D, T=T, L=L, doctype='tset', indices=kargs.get('docIDs', [])) # other params: cohort, seq_ptype

        # by default, training data document is not saved (to allow for dynamically generated training set)
        outputdir = TSet.getPath(cohort=kargs.get('cohort', None), dir_type=kargs.get('dir_type', 'train')) # see seqparams.TSet.getPath
        if kargs.get('save_', False): 

            # output directory may depend on cohort, (which if not given, a System.cohort will be used)
            # getPath will create a new directory by default if not already existed 
            # outputdir = TSet.getPath(cohort=kargs.get('cohort', None), dir_type=kargs.get('dir_type', 'train')) # see seqparams.TSet.getPath
            if kargs.has_key('outputdir'): outputdir = kargs['outputdir'] # overwrite default 
            assert os.path.exists(outputdir), "Invalid training set output directory: %s" % outputdir

            fname = kargs.get('outputfile', None)
            if fname is None: 
                # index: used to distinguish among multiple training and test sets
                # suffix: served as secondary ID (e.g. training data made from sequences of different contents, tset with labeling='lcs')
                fname = TSet.getName(cohort=cohort_name, d2v_method=d2v_method, index=kargs.get('index', None), 
                                        seq_ptype=seq_ptype, suffix=suffix, 
                                        file_stem='mcs')  # file_stem <- 'mcs' instead of 'tset'

            fpath = os.path.join(outputdir, fname)
            df.to_csv(fpath, sep=TDoc.csv_sep, index=False, header=True)  # '|'

        return df

### end class TDoc(seqparams.TDoc)


def docToCSV(D, T=[], L=[], **kargs): 
    """
    Save (transformed) MCSs to dataframe (.csv) format. 

    Each patient corresponds to a single document. 


    Input 
    ----- 
    cohort
    seq_ptype

    sequences (D)
    timestamps (T)
    labels (L)

    
    optional: 
        outputdir
        ifiles
        indices: document IDs (useful when documents are segmented)

    Output 
    ------
    dataframe containing following columns 
        
        ['sequence', 'timestamp', 'label', ]

        where 'timestamp' and 'lable' are only included when available


    Memo
    ----
    1. headers of source files used to make sequence documents
            header_condition = ['person_id', 'condition_start_date', 'condition_source_value', ]
            header_drug = ['person_id', 'drug_exposure_start_date', 'drug_source_value', ]
            header_lab = ['person_id', 'measurement_date', 'measurement_time', 'value_as_number', 'value_source_value']

    2. This is derived from seqReader.readDocToCSV()

    """
    def seq_to_str(seqx, sep=','): 
        # must be a list/array/sequence of sequence; 95% a list of lists
        assert hasattr(seqx, '__iter__'), "Invalid documents:\n%s\n" % seqx
        if len(seqx) == 0: 
            print('docToCSV> Warning: Empty doc!')
            return [] 

        rp = random.randint(0, len(seqx))
        if not isinstance(seqx[rp], list): 
            print('docToCSV> Warning: Input D is not a list of lists (class labels?) Example:\n%s\n' % seqx[rp])
            seqx2 = []
            for item in seqx: 
                if hasattr(item, '__iter__'): 
                    seqx2.append(sep.join(str(e) for e in item))
                elif isinstance(item, str):
                    seqx2.append(item) # do nothing 
                else: 
                    seqx2.append(str(item))  # e.g. integers as labels
            return seqx2
        return [sep.join(str(tok) for tok in doc) for doc in seqx] 

    def str_to_seq(df, col='sequence', sep=','):
        seqx = []
        for seqstr in df[col].values: 
            s = seqstr.split(sep)
            seqx.append(s)
        return seqx
    def get_cohort(): 
        try: 
            return kargs['cohort']
        except: 
            pass 
        raise ValueError, "cohort info is mandatory."
    def add_doc_ids(adict, test_ids=[]):  # <- D
        docIDs = kargs.get('indices', [])
        if len(docIDs) > 0: 
            assert len(docIDs) == len(D), "length inconsistency (%d vs nD=%d)" % (len(docIDs), len(D))
            adict['index'] = docIDs
        else: 
            adict['index'] = range(0, len(D))
        return
    def add_mcs(adict, test_ids=[]):  # <- D 
        adict['sequence'] = seq_to_str(D)   # assuming visit-seperator ';' is lost 

        if test_ids:
            for r in test_ids:  
                print('  + example D:\n%s\n' % adict['sequence'][r])
        return
    def add_times(adict, test_ids=[]): # T
        if len(T) > 0: 
            adict['timestamp'] = seq_to_str(T)
            if test_ids:
                for r in test_ids:  
                    print('  + example T:\n%s\n' % adict['timestamp'][r])
        else: 
            pass # noop
        return
    def add_labels(adict, test_ids=[]): # L
        if len(L) > 0: 
            adict['label'] = seq_to_str(L) 
            if test_ids:
                for r in test_ids:  
                    print('  + example L:\n%s\n' % adict['label'][r])
        else: 
            pass # noop 
        return 
    def ordering(adict, attributes=[]): 
        if not attributes: attributes = ['index', 'sequence', 'timestamp', 'label', ]
        return [a for a in attributes if a in adict]
        
    import random

    # first, read the docs from source files (.dat)
    doctype = kargs.get('doctype', 'tset') # other types defined in base seqparams.TDoc: {'timed', 'labeled'}
    if len(T) > 0: assert len(D) == len(T)
    print('info> nD: %d (=?= nT: %d), doctype: %s, n_labels: %d' % (len(D), len(T), doctype, len(L)))

    # [params] IO 
    ret = {}  # output
    # basedir = kargs.get('outputdir', sys_config.read('DataExpRoot'))  # sys_config.read('DataIn') # document in data-in shows one-visit per line
    # fname = seqparams.TDoc.getName(cohort=kargs['cohort'], doctype=doctype, ext='csv') 
    seq_ptype = kargs.get('seq_ptype', 'regular') # if regular, then doc_basename is set to TDoc's default: condition_drug

    # A. fpath convention for source MCS file
    # fpath = TDoc.getPath(cohort=get_cohort(), seq_ptype=seq_ptype, doctype=doctype, ext='csv', baesdir=basedir)  # get the path to the new document

    # B. fpath convention for derived MCS file || training set (see TSet.saveDoc(), TDoc.saveDoc())
    # see TDoc.save()

    df = DataFrame() # dummy 
    attributes = ['index', 'sequence', 'timestamp', 'label', ] 
    nD = len(D)
    if nD > 0: # this is big! ~1.2G 
        
        # [test]
        r = random.randint(0, nD)
        # pickle.dump(sequences, open(fpath, "wb" ))

        adict = {} # {h: [] for h in header}
        if doctype in ['tset', ]:
             
            add_doc_ids(adict)  # adict['index'] 
            add_mcs(adict, test_ids=[r, ])
            
            add_times(adict, test_ids=[r, ])
            add_labels(adict, test_ids=[r, ])
            header = ordering(adict, attributes)  # ensure a specified order

            df = DataFrame(adict, columns=header)
            # df.to_csv(fpath, sep='|', index=False, header=True)
        else: # doctype <- 'doc', 'visit'
            attributes = ['index', 'sequence', ]   # seqMaker2 now also can be configued to produce this (but without seq_ptype as part of the file ID)
            
            add_doc_ids(adict)  # adict['index']
            add_mcs(adict, test_ids=[r, ])     
            header = ordering(adict, attributes)

            df = DataFrame(adict, columns=header)
            # df.to_csv(fpath, sep='|', index=False, header=True)

        print('docToCSV> Got (transformed) MCSs (size: %d, doctype: %s, dim: %s)' % (len(D), doctype, str(df.shape)))
    else: 
        raise ValueError, "Empty input documents (D)!"

    return df

def toSingleLabel(labelx, pos=0): 
    from labeling import TDocTag
    return TDocTag.toSingleLabel(labelx, pos=pos)  # pos: position of the desired label in multilabel format

def getSurrogateLabels(docs, **kargs): 
    """
    Determine class labels (cohort-specific) for the input documents (docs). 
    Note that the class label are NOT the same as the document label used for Doc2Vec. 
    If the input cohort is not recognized, this returns a uniform positive labels by default 
    i.e. each document is 'positive (1)'

    Params
    ------
    cohort
    """
    import labeling 
    return labeling.getSurrogateLabels(docs, **kargs)

def readDocFromCSV(**kargs):  # ported from seqReader
    """
    Read coding sequences from .csv files 

    Input
    -----
    cohort: name of the cohort (e.g. PTSD) ... must be provided; no default
            used to figure out file names
    ifiles: sources of coding sequences 
            if ifiles are given, the cohort is igonored 
    (o) basedir: directory from which sources are stored (used when ifiles do not include 'prefixes')

    Memo
    ----
    import seqReader as sr 
    sr.readDocFromCSV(**kargs)
    
    """
    def get_sources(complete=True):
        ifiles = kargs.get('ifiles', [])
        if not ifiles: assert cohort_name is not None

        # [note] doctype has to be 'timed' because timestamps are expected to be in the .csv file
        if complete and not ifiles: # cohort-specific
            for doctype in ['labeled', 'timed', 'doc', 'visit', ]:  # the more complete (more meta data), the higher the precedence
                docfiles = TDoc.getPaths(cohort=cohort_name, basedir=docSrcDir, doctype=doctype, 
                    ifiles=[], ext='csv', verfiy_=True)  # if ifiles is given, then cohort is ignored
                if len(docfiles) > 0: break
        else: # doctpye specific (or ifiles are given)
            # [use] 
            # 1. ifiles <- a list of file names (of the source documents) then use basedir to figure out full paths
            # 2. ifiles <- full paths to the document sources (including prefixes)
            docfiles = TDoc.getPaths(cohort='n/a', basedir=docSrcDir, doctype=kargs.get('doctype', 'timed'), 
                ifiles=ifiles, ext='csv', verfiy_=True)  # if ifiles is given, then cohort is ignored
        assert len(docfiles) > 0, "No %s-specific sources (.dat) are available under:\n%s\n" % (cohort_name, docSrcDir)
        return docfiles   
    def str_to_seq(df, col='sequence', sep=','):
        seqx = [] # a list of (lists of tokens)
        for seqstr in df[col].values: 
            tokens = seqstr.split(sep)
            seqx.append(tokens)
        return seqx

    header = ['sequence', 'timestamp', 'label', ] 
    listOfTokensFormat = TDoc.fListOfTokens # ['sequence', 'timestamp', ]

    cohort_name = kargs.get('cohort', None)  # needed if ifiles is not provided
    basedir = docSrcDir = kargs.get('inputdir', sys_config.read('DataExpRoot'))  # sys_config.read('DataIn') # document in data-in shows one-visit per line

    # if 'complete' is True => try the most information-rich format first i.e. in the order of doctype='labeled', 'timed', 'doc', 'visit'
    fpaths = get_sources(complete=kargs.get('complete', True)) # [params] cohort, basedir
    print('read> reading from %d source files:\n%s\n' % (len(fpaths), fpaths))

    # [output]
    ret = {h: [] for h in header}
    # sequences, timestamps, labels = [], [], []

    # [policy] if there are multiple sources, their contents will be consolidated
    for fpath in fpaths: 
        df_seq = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
        if not df_seq.empty: 
            for h in header: 
                if h in header: 
                    # or use seqparams.TDoc.strToSeq(df_seq, col='sequence')
                    ret[h].extend(str_to_seq(df_seq, col=h, sep=','))  # assuming that ';', '$' are all being replaced

    assert len(ret) > 0, "No data found using the given attributes: %s (columns expected: %s)" % (header, df_seq.columns.values)
    return ret # keys <- header

def analyzePrediagnosticSequence(codes, **kargs):
    return apseq(codes, **kargs) 
def apseq(codes, **kargs):
    def normalize_input(x): 
        if isinstance(x, str): 
            x = x.split(kargs.get('sep', ' '))
        assert hasattr(x, '__iter__'), "Invalid input codes: %s" % x
        return x # a list of code strings
    def to_str(seq, sep=','): 
        return sep.join(str(e) for e in seq)
    def code_str(seq, sep='-'): # convert codes to file naming friendly format 
        s = to_str(seq, sep=sep) 
        # alternative: s = s.translate(None, string.punctuation)  # remove punctuations '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        return s.replace('.', '')
    def time_to_diagnoses(sid, test_=False): # [params] tseqx, matched_index_set
        # use sid to index timestamps accordingly 
        if not tseqx: 
            print('time_to_diagnoses> no input timestamps > aborting ...')
            return None
        
        ret = {} # keys: ['time_to_first', 'time_to_last', 'days_elapsed', 'n_visits'] 

        # datetime.strptime("2017-08-11", "%Y-%m-%d")
        tseq = tseqx[sid]  # corresponds to the sid-th sequence
        t0 = datetime.strptime(tseq[0], "%Y-%m-%d")  # datetime object ready for arithmetic
        tseq_arr = np.array(tseq)
        tlist = []
        for idx in matched_index_set: 
            times = tuple(tseq_arr[idx])

            # keys: occurrence of the first code; values: occurrences of all input codes
            td0 = times[0]
            # tdict[t0] = times
            tlist.append((td0, times))  # times in string format

        tlist = sorted(tlist, key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))
        if test_: print('  + sorted times:\n%s\n' % tlist[:10])

        # time to first diagnosis
        td0 = datetime.strptime(tlist[0][0], "%Y-%m-%d") # first observed, first code
        tdf = datetime.strptime(tlist[-1][0], "%Y-%m-%d") # last observed, first code
        ret['time_to_first'] = (td0-t0).days   # (td0 - t0) ~> datetime.timedelta object
        ret['time_to_last'] = (tdf-t0).days
        ret['n_visits'] = len(dict(tlist))  # constraint: different dates
        ret['n_total_visits'] = len(tlist)  
        if test_: assert ret['n_total_visits'] >= ret['n_visits']
        ret['days_elapsed'] = ret['time_to_last']-ret['time_to_first']
        assert ret['days_elapsed'] >= 0

        # all questions answered? 
        if test_: assert len(set(time_dependent_fields)-set(ret.keys()))==0

        return ret
    def has_precodes(seq): # any of the prediagnostic codes occur in the pre-diag sequences (i.e. sequence up to first occurence of target)
        # precodes = kargs.get('precodes', [])
        if not precodes: 
            print('  + Warning: No pre-diagnostic codes given => set to False.')
            return False 
        
        n0 = len(set(precodes))
        n1 = len(set(precodes)-set(seq))
        return n0 > n1  # then a subset of precodes must be in 'seq'
    def time_precode_target(seq, tseq, tid=None): # sid: the sid-th document, need this to find corresponding timestamp
        assert len(seq) == len(tseq)
        if tid is None: tid = sd00  # first match (of target), first code
        # time elapsed (days) between first relevant pre-diagnostic code to the target
        i0, c0 = 0, '000.0'
        precode_set = set(precodes)
        for i, c in enumerate(seq): 
            if c in precode_set: 
                i0 = i; c0 = c
                break

        # time 
        assert tid > i0, "Target: %s must occur after pre-diag code: %s" % (codes[0], c0)
        t0 = datetime.strptime(tseq[i0], "%Y-%m-%d")  # datetime object for the first occurrence of any of the precodes
        tF = datetime.strptime(tseq[tid], "%Y-%m-%d")
        delta = (tF-t0).days
        return delta
    def eval_common_codes_prior(seq): # the most common codes before target
        commonCodesPrior.update(seq)
        return
    def eval_common_ngrams_prior(seq, max_length=4): # [note] cannot just operate on 'seqx' becaues each 'seq' may be different
        # length -> Counter: {(ngr, count)}
        commonNGramsPrior.update(seqAlgo.count_ngrams2([seq, ], min_length=1, max_length=max_length, partial_order=False)) # length => {(ngr, count)}
        return 
    def test_matched(seq, seq0, seqF, ithmatch): 
        print('  + this is the %d-th matches so far ...' % ithmatch)
        print('  + example seq (prior to FIRST mention, n=%d):\n%s\n' % (len(seq0), seq0))
        print('  + example seq (prior to LAST mention, n=%d):\n%s\n' % (len(seqF), seqF))
        print('  + lengths (to first mention): %d vs (to last): %d' % (len(pre0_seq), len(preF_seq)))
        return
    def summary_report(topn=10, max_length=4): # has to be the last statement of the outer function 
        n_total = sum(n_persons_batch)
        n_total_precodes = sum(n_persons_precodes_batch)
        n_total_precodes_last = sum(n_precodes_last_batch)
        r_precodes = n_total_precodes/(n_total+0.0)
        r_precodes_last = n_total_precodes_last/(n_total+0.0)

        print('tdoc.analyzer> Found %d eligible persons (grand total: %d) with target codes: %s' % (n_total, nGrandTotal, to_str(codes)))
        print('               + among all eligible, %d (r=%f) of them has precodes (prior to FIRST mention)' % (n_total_precodes, r_precodes))
        print('               + among all eligible, %d (r=%f) of them has precodes (prior to LAST mention)' % \
            (n_total_precodes_last, r_precodes_last))

        # most common codes 
        topn_codes = commonCodesPrior.most_common(topn)
        print('               ++ Top %d codes:\n%s\n' % (topn, topn_codes))

        # most common bigrams 
        # for i in range(1, max_length+1): 
        for length in [2, ]:  
            topn_ngrams = commonNGramsPrior[length].most_common(topn)
            print('               ++ Top %d %d-grams:\n%s\n' % (topn, length, topn_ngrams))
        return

    global gHasConfig   # an attempt to run this module without config package
    import collections
    # import seqAlgo
    # import seqReader as sr 
    # from datetime import datetime
    # import time 
    # import pandas as pd
    precodes = kargs.get('precodes', [])
    print('input> target codes:\n%s\ninput> prediagnostic codes (e.g. BP prior to SpA):\n%s\n' % (codes, precodes))

    docSources = kargs.get('ifiles', [])
    cohort_name = kargs['cohort'] if not docSources else 'n/a' # must provide cohort (to determine file names of the document sources)
    basedir = os.path.join(sys_config.read('DataExpRoot'), 'sequencing') if gHasConfig else os.getcwd()
    if not docSources: 
        docSources = TDoc.getPathsByCohort(cohort)  # source documents are all in default dir: tpheno/data-exp 

    # [note] in order to use readDocFromCSV, need to create .csv from .dat first (see sr.readDocToCSV())
    # ret = readDocFromCSV(cohort=cohort_name, ifiles=docSources, basedir=basedir)
    codes = normalize_input(codes)

    # [output]
    header = ['target', 'time_to_first', 'latency', 'n_visits', 'sequence_to_first', 'has_precodes', 'time_to_last', 'days_elapsed', ]   # time_to_diagnosis: really is time to the FIRST diagnosis
    adict = {h:[] for h in header} # [output]
    # condition_drug_seq-group-N.dat where N = 1 ~ 10 
    # [note] query document source one by one since they are large
    nD = 10
    allow_partial_match = False
    n_total_matched = 0
    time_dependent_fields = ['time_to_first', 'time_to_last', 'days_elapsed', 'n_visits', ]  # answered by time_to_diagoses operation
    nGrandTotal = 0  # number of perons in the entire sequenced data (DB)
    n_persons_batch, n_persons_precodes_batch, n_precodes_last_batch = [], [], []
    lenToFirst_precodes_distr, lenToLast_precodes_distr = [], []
    commonCodesPrior = collections.Counter()
    commonNGramsPrior = {}  # length -> Counter: {(ngr, count)}
    for ithdoc, fpath in enumerate(docSources):
        # ret = TDoc.loadGeneric(filenames=[fname, ]) # prefix: tpheno/data-exp/sequencing

        # normalize input path
        prefix, fname = os.path.dirname(fpath), os.path.basename(fpath) 
        if prefix is None: prefix = basedir # default is to analyze sequencing dir 
        fpath = os.path.join(prefix, fname)
        assert os.path.exists(fpath), "Invalid path: %s" % fpath
                
        # read timed coding sequences
        # [note] load + parse vs TDoc.loadGeneric() does not include parsing (coding sequences require some interpretations)
        #        if no_throw <- True, then skip visit segments with parsing errors (i.e. cannot separate time)
        seqx, tseqx = sr.readTimedDocPerPatient(cohort='n/a', ifiles=[fname, ], inputdir=basedir, no_throw=True) 
        assert not tseqx or len(seqx) == len(tseqx)

        # hasLabel = True if len(labels) > 0 else False
        div(message='Found %d sequences in %d-th src: %s | has timestamp? %s,' % (len(seqx), ithdoc, fname, len(tseqx)>0))
        nGrandTotal += len(seqx)
        # analyze this file    
        # n_persons: number of persons matching target codes 
        # n_persons_precodes: matched targets and contain predignositc codes (prior to first mention)
        # n_precodes_last: matched targets and contain pre-diag codes (prior to LAST mention)
        n_persons = n_persons_precodes = n_precodes_last = 0  
        for i, seq in enumerate(seqx): 
            q, r = codes, seq
            if i < 10: assert isinstance(r, list)

            # is it matched? matched positions 
            matched_index_set = seqAlgo.traceSubsequence3(q, r)
            tMatched = True if len(matched_index_set) > 0 else False            

            # find the time of the first mention (of the input coding seqments, ordering important)
            # number of occurrences (of input codes)
            # compute time to diagnosis
            # number of mentions corresponding to different dates/visits (assumption)

            if tMatched: # matched => found target/input codes in the sequence
                sd00 = matched_index_set[0][0] # position of first match (in the entire sequence), index of the first code in codes 
                sf = matched_index_set[-1][0] # last match, first code

                subseq = to_str(seq[:sd00+1])

                # query ['time_to_first', 'time_to_last', 'days_elapsed', 'n_visits']
                tdict = time_to_diagnoses(sid=i, test_=(i % 5==0)); assert tdict is not None, "No timestamps."
                adict['target'].append(to_str(codes)) 
                adict['sequence_to_first'].append(subseq)
                for h in time_dependent_fields: 
                    adict[h].append(tdict[h])
                n_persons += 1  # per source
                n_total_matched += 1  # overall 

                # [query] pre-diagnostic codes analysis
                pre0_seq = seq[:sd00]; lenToFirst_precodes_distr.append(len(pre0_seq)) # [test]
                if has_precodes(pre0_seq):  # params: kargs['precodes'] 
                    adict['has_precodes'].append(1)
                    n_persons_precodes += 1 
                    
                    # [query] time elapsed from first precode to target? 
                    delta = time_precode_target(seq, tseq=tseqx[i], tid=sd00) # seq, timestamp, tid: target position
                    adict['latency'].append(delta)  # unit: days

                else: 
                    adict['has_precodes'].append(0)
                    adict['latency'].append(-1) # this will be removed
                    
                preF_seq = seq[:sf]; lenToLast_precodes_distr.append(len(preF_seq)) # [test]
                if has_precodes(preF_seq):
                    n_precodes_last += 1    # number of persons | matched + has pre-diagnostic codes before last mention (of target) 

                # most common codes prior to targets
                # eval_common_codes_prior(pre0_seq) # updates commonCodesPrior 714.0, 715.0
                commonCodesPrior.update(pre0_seq) # don't count target

                # most common n-grams prior to targets
                eval_common_ngrams_prior(pre0_seq, max_length=4)
                    
                # [test]
                if n_total_matched % 5 == 0: 
                    test_matched(seq, seq0=pre0_seq, seqF=preF_seq, ithmatch=n_total_matched)
        ### end foreach sequence in 'ithdoc'-th source
        
        # collect number-of-matches-statistics 
        n_persons_batch.append(n_persons)  
        n_persons_precodes_batch.append(n_persons_precodes) 
        n_precodes_last_batch.append(n_precodes_last)
        div(message='End of %d-th sequencing file: Found %d candidates, in which %d contain precodes.' % (ithdoc, n_persons, n_persons_precodes))

    # save report 
    df = DataFrame(adict, columns=header); N0 = df.shape[0]; assert N0 == sum(n_persons_batch)
    
    # save only those with precodes (prior to first mention)? 
    # [filter]
    df = df.loc[df['has_precodes']==1]; NHasPC = df.shape[0]; assert NHasPC == sum(n_persons_precodes_batch)
    df.drop(['has_precodes'], axis=1, inplace=True)
    print("  + n_persons: %d => n_persons(has precodes): %d" % (N0, NHasPC)) 

    # [note] tpheno/seqmaker/data/SpA/prediag_analysis_C7200.csv
    fpath = os.path.join(seqparams.getCohortDir(cohort='SpA'), 'prediag_analysis_C%s.csv' % code_str(codes, sep='-'))
    df.to_csv(fpath, sep='|', index=False, header=True)  
    print("  + IO: saved summary to:\n%s\n" % fpath)
    
    # total persons found 
    assert len(n_persons_batch) == len(docSources)
    summary_report() # has to be the last statement
                
    return 

def t_subsequence(**kargs):
    def get_sequencing_files(n=10): 
        fpat = 'condition_drug_timed_seq-group-%d.dat'
        files = []
        for i in range(1, n+1):
            # files.append(fpat % i)
            # return files
            yield os.path.join(basedir, fpat % i)
    def get_bp_codes(): 
        bp_codes = ['724', '724.0', '724.00', '724.01', 724.02, 724.03, 724.09, 724.1, 724.2, 724.3, 724.4, 724.5, 724.6, 724.8, 724.9]
        return [str(e) for e in bp_codes]

    import seqAlgo
    basedir = os.path.join(sys_config.read('DataExpRoot'), 'sequencing') if gHasConfig else os.getcwd()

    bp_codes = get_bp_codes()
    target_codes = ['720.0', ]

    # analyze 720.0, SaP
    apseq(target_codes, precodes=bp_codes, ifiles=list(get_sequencing_files()), cohort='n/a')

    return

def test(**kargs): 
    t_subsequence(**kargs) 

if __name__ == "__main__": 
    test()
