# encoding: utf-8

import os, sys, re

from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
# from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

# probability calibration
from sklearn.calibration import CalibratedClassifierCV

import numpy as np
import multiprocessing

from config import sys_config
from batchpheno import dfUtils
from batchpheno.utils import div

# d2v via gensim
from gensim.models import Doc2Vec
import gensim.models.doc2vec

### Global Definitions 
token_sep = ','
visit_sep = ';'
token_end_history =  '$'
doc_sep = patient_sep = '\n'

lsep = '_'   # label separator
generic_feature_prefix = 'f'

# utility  [todo] port to systerm
def format_list(alist, mode='h', sep=', ', n_pad_space=0):  # horizontally (h) or vertially (v) display a list 
    if mode == 'h': 
        s = sep.join([e for e in alist])  
    else: 
        s = ''
        spaces = ' ' * n_pad_space
        for e in alist: 
            s += '%s%s\n' % (spaces, e)
    return s

# global parameters 
class System(object): 
    cohort = 'generic'  # 'diabetes', 'PTSD'
    # seq_ptype = 'regular'  # sequence content type for training data

    label_map = {}  # see an example in relabel()

    @staticmethod
    def relabel(lmap={}):  # this definition may not be ths same as lcsHandler.relabel()
        # CKD stage classes
        if not lmap: 
            lmap['Control'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
            lmap['CKD Stage 5'] = ['ESRD after transplant', 'ESRD on dialysis', ]
            lmap['CKD Stage 3'] = ['CKD Stage 3a', 'CKD Stage 3b', ]

            print('System.relabel> Applying default map with %d entries.' % len(lmap))
            for k, v in lmap.items(): 
                print('  + %s => %s' % (k, format_list(v, mode='h')))

        System.label_map = lmap
        return lmap
    @staticmethod
    def convert_labels(labels, lmap={}, inplace=False, unique=False):
        if not lmap: 
            lmap = System.relabel()  # use default mapping 
        
        lookup = {k:k for k in lmap.keys()}
        for ml, label_set in lmap.items():  # ml: mapped label
            for l in label_set: 
                if not lookup.has_key(l): 
                    lookup[l] = ml  # map l to ml
                else: 
                    # should not have mapped 
                    # raise ValueError, "Label %s has been mapped to %s" % (l, lookup[l])
                    pass 

        if inplace: 
            for i, label in enumerate(labels): 
                labels[i] = lookup.get(label, label)
        else: 
            lp = []
            for label in labels: 
                lp.append(lookup.get(label, label))
            labels = lp

        if unique:
            # labels = np.unique(labels) # this does not preserve order

            # preserve the order
            label_set = set() 
            lp = []
            for label in labels: 
                if label in label_set: continue
                
                lp.append(label)
                label_set.add(label)
            labels = lp
           
        return labels 

### end class System

class W2V(object): 
    """
    
    Params
    ------
    window: is the maximum distance between the predicted word and context words used for prediction within a document.
    min_count: ignore all words with total frequency lower than this.

    """
    n_features = 50
    window = 5
    min_count = 2  # set to >=2, so that symbols that only occur in one doc don't count
    
    n_iter = 25 # default: 5, # corresponds to 'iter' in gensim.models.doc2vec.Doc2Vec
    n_cores = multiprocessing.cpu_count()
    # print('info> number of cores: %d' % n_cores)
    n_workers = max(n_cores-20, 18)

    # word2vec_method = 'SG'  # skipgram 
    # doc2vec_method = 'PVDM'  # default: distributed memory
    w2v_method = 'sg'  # options: 'cbow'
    read_mode = 'doc' 

    @staticmethod
    def show_params(): 
        div(message='Word2Vec Parameters', symbol='%')
        msg  = '  + current w2v method: %s\n' % W2V.w2v_method 
        msg += '  + number of features: %d\n' % W2V.n_features
        msg += '  + window size: %d\n' % W2V.window 
        msg += '  + ignore tokens with total freq less than %d\n' % W2V.min_count
        msg += '  + number of epochs: %d\n' % W2V.n_iter

        print msg 

### end class W2V 

class D2V(W2V): 
    """

    Params
    ------
    dm: defines the training algorithm. By default (dm=1), 'distributed memory' (PV-DM) is used. 
        Otherwise, distributed bag of words (PV-DBOW) is employed.

    dm_mean: if 0 (default), use the sum of the context word vectors. 
             If 1, use the mean. Only applies when dm is used in non-concatenative mode.


    (*) choosing between hierarchical softmax and negative sampling 

    hs = if 1, hierarchical softmax will be used for model training. 
         If set to 0 (default), and negative is non-zero, negative sampling will be used.

        (*) how many negative samples? 

            negative:  if > 0, negative sampling will be used, the int for negative specifies how many 
                       “noise words” should be drawn (usually between 5-20). Default is 5. 
                       If set to 0, no negative samping is used.

    References 
    ----------
    1. gensim
       https://radimrehurek.com/gensim/models/doc2vec.html
    2. 
    """
    # W2V parameters 

    # number of iterations (epochs) over the corpus. The default inherited from Word2Vec is 5, 
    # but values of 10 or 20 are common in published ‘Paragraph Vector’ experiments.
    n_iter = 20  # corresponds to 'iter' in gensim.models.doc2vec.Doc2Vec

    n_features = 100
    window = 5
    min_count = 2  # set to >=2, so that symbols that only occur in one doc don't count

    dm = 1 

    # if 1, then use concatenation of context vectors rather than sum/average
    dm_concat = 0

    # [condition] dm_concat == 0, not in concatentation mode 
    # if 0 (default), use the sum of the context word vectors. If 1, use the mean; it applies only when dm_concat <- 0
    dm_mean = 1  

    # dbow_words if set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training; 
    # default is 0 (faster training of doc-vectors only).
    dbow_words = 0 

    # this may depend on V (e.g. large CKD cohort has 20654206 or 20M+ unique tokens)
    negative = 15 # [condition] used when hs <- 0
    hs = 0  # negative sampling 

    # overwrite this value for other approximation methods (e.g. controlled negative sampling)
    prob_approx_method = 'negative sampling' if hs == 0 else 'hierarchical softmax'  # default

    supported_methods = ['pv-dm2', 'pv-dm', 'pv-dbow', 'tf-idf', ]  # [note] pv-dm2 = pv-dm + pv-dbow
    d2v_method = 'pv-dm2'  # options: 'pv-dmdbow', 'pv-dm', 'pv-dbow', 'tf-idf', 'external' (embedding is done externally), 'bow'

    # document label attributes 
    label_attributes = ['words', 'tags', ]  # default attributes used in gensim
    
    # example models
    simple_models = [
        # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
        Doc2Vec(dm=1, dm_concat=1, size=100, window=W2V.window, negative=5, hs=0, min_count=W2V.min_count, workers=W2V.n_cores),
        # PV-DBOW 
        Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=W2V.min_count, workers=W2V.n_cores),
        # PV-DM w/average
        Doc2Vec(dm=1, dm_mean=1, size=100, window=W2V.window, negative=5, hs=0, min_count=W2V.min_count, workers=W2V.n_cores),
    ]

    @staticmethod
    def show_params(): 
        div(message='Doc2Vec Parameters', symbol='%')
        msg  = '  + current d2v method: %s\n' % D2V.d2v_method 
        msg += '  + number of features: %d\n' % D2V.n_features
        msg += '  + window size: %d\n' % D2V.window 
        msg += '  + ignore tokens with total freq less than %d\n' % D2V.min_count
        msg += '  + number of epochs: %d\n' % D2V.n_iter
        
        prob_approx_method = D2V.prob_approx_method

        msg += '  + training method: %s' % prob_approx_method
        if D2V.hs == 0: 
            msg += '  + number of negative samples: %d' % D2V.negative

        print msg 

### end class D2V

class TDoc(object): # class Temporal Document, TDoc
    """
    Medical records of a patient typically contain diagnosis and treatment information, which when expressed 
    in terms of the medical coding system translates into a list of diagnostic codes, medication codes, and lab codes, among others. 

    Suppose for simplicity, we only consider diagnostic and medication codes
    then one can organize patients' medical history using timestamps in DB 

    Thus, the entire medical history consists of temporally ordered medical visists => a set of visits 
    where each visis is associated with a set of codes

    Params
    ------
    read_mode: 'doc', 'visit', 'timed'
        'doc' one patient corresponds to one document by concatenating all visits (each of which is associated with a set of codes)
        'visit': one patient corresponds to multiple documents, each of which represnets a single clinical visist. 


    """ 
    doctype = read_mode = 'doc' # doc_types
    doc_basename = 'condition_drug'
    doc_types = ['doc', 'timed', 'labeled', ]  # obsolete: 'visit', per-visit-per-doc format (which can be derived from 'doc' format)

    tdoc_prefix = 'condition_drug'  # this tells us what types of medical codings are included in the document; serves part of the file naming
    f_base = 'seq'  # 

    ### Docuement sources, directories, etc. 
    prefix = sys_config.read('DataExpRoot')
    prefix_sequencing = os.path.join(prefix, 'sequencing') # example files: condition_occurrence-0.csv, condition_occurrence-all.csv

    ### Syntax 
    
    # delimiters  
    token_sep = ','
    token_end_visit = ';' # each visit can consists of multiple diagnoses and medications
    token_end_history = '$'  # end token of entire medical history (of a patient)
    token_unknown = 'unknown'
    token_empty = 'empty'
    patient_sep = '\n'  # separator for different patient documents; this is to facilitate IO
    sep_csv = '|'

    # formatting in .csv (i.e. structured coding sequence files) 
    fListOfTokens = ['sequence', 'timestamp', ]   
    fLabel = ['label', ] # backward compatibility

    # legit class label types
    fLabels = ['label', 'label_lcs', 'label_unigram', 'label_bigram', 'label_trigram', 'label_4gram', ]  
    label_unknown = 'unknown'  # e.g. used for empty documents or any documents under exclusion 
    time_unknown = '9999-01-01'

    # document labels
    doc_label_prefix = 'client'  # use the definition in labeling.DocTag instead

    # system
    sequencing_subdir = 'sequencing'  # subdirectory specific for "sequencing data" 

    # schema
    header_timed_seq = ['sequence', 'timestamp', ]
    header_labeled_seq = ['sequence', 'timestamp', 'label']

    ### Grammar

    ### Protocol

    @staticmethod
    def getSourceTable(**kargs): # intermmediate file (person_id, start_date, concept_id)
        """
        Get the intermediate dataframe with the header (person_id, start_date, concept_id/source_value)
        from which source documents (of coding sequences) are composed

        Params
        ------
        identifier
        index

        Memo
        ----

        Use 
        ---
        * seqMaker2, seqMaker3 
        * cohort

        """
        import pandas as pd
        from config import sys_config
        fsep = '|'
        header = kargs.get('header', ['person_id', 'start_date', 'concept_id'])
        index = kargs.get('index', None)
        identifier = kargs.get('identifier', TDoc.doc_basename) # condition_drug by default
        if index is not None: identifier = "%s-%s" % (identifier, index)
        outputdir = kargs.get('outputdir', TDoc.prefix_sequencing)  # default: tpheno/data-exp/sequencing
        fpath = os.path.join(outputdir, "%s.csv" % identifier)
        assert os.path.exists(fpath), "getSourceTable> Invalid path: %s" % fpath
        # df.to_csv(fpath, sep=fsep, index=False, header=True, encoding='utf-8')
        # print('io> Saved result set (dim=%s) to %s' % (str(df.shape), fpath))  
        df = pd.read_csv(fpath, sep=fsep, header=0, index_col=False, error_bad_lines=True)
        assert set(header).issubset(df.columns), \
            "getSourceTable> a subset of header does not exist: %s is not subset of %s" % (header, df.columns.values)
        return df[header]

    @staticmethod
    def getTDocPrefix(content=None):
        if content:
            raise NotImplementedError 
        return TDoc.tdoc_prefix

    @staticmethod
    def setName(cohort, **kargs): 
        """

        Params
        ------
        cohort: 
        seq_compo:

        include_timestamps

        meta: additional information tagged to the file name
              e.g. if the document contains only 'med', then set meta to 'med'
              => condition_drug_seq-PTSD-med

        Todo
        ---- 
        .id seems to have special meanings on Windows

        """
        cohort_name = cohort
        include_timestamps = kargs.get('include_timestamps', False)

        # [params] time series document settings
        seq_compo = seq_composition = kargs.get('seq_compo', None)
        if seq_compo is not None: TDoc.tdoc_prefix = seq_compo

        # [params] output file 
        # [note] default cohort if None is diabetes; other cohorts e.g. PDSD cohort: condition_drug_seq-PTSD.dat
        # f_base = 'timed_seq' if include_timestamps else 'seq'

        # determine the file stem (i.e. f_base)
        doctype = TDoc.doctype  # default: 'doc'
        if include_timestamps: doctype = 'timed'
        f_base = TDoc.doctypeToStem(doctype=doctype)  # doc, timed, labeled
        if cohort_name is not None: f_base = '%s-%s' % (f_base, cohort_name)
        meta = kargs.get('meta', None)
        if meta is not None: f_base = '%s-%s' % (f_base, meta)  # e.g.  condition_drug_seq-PTSD-med

        ret = {}
        ret['dat'] = f_tdoc = '%s_%s.dat' % (TDoc.tdoc_prefix, f_base)  # [output]
    
        # .csv contains a structured format of the same coding sequences (whereas .dat only contains the document)
        # structured format provides more infomration (meta data) such as the timestamp of each code and the document label
        ret['csv'] = f_tdoc_csv = '%s_%s.csv' % (TDoc.tdoc_prefix, f_base)  # applies for doctype in {'timed', 'labele', }
    
        # only one kind of id file
        # [todo]
        ret['id'] = f_tdoc_id = '%s_%s.id' % (TDoc.tdoc_prefix, f_base) 
        print('io> .dat file name: %s, .csv: %s, .id: %s' % (f_tdoc, f_tdoc_csv, f_tdoc_id))

        return ret # keys: ['dat', 'csv', 'id', ]
    @staticmethod
    def getIDFile(cohort, **kargs):
        # [todo]
        ret = setName(cohort, **kargs)  # exmaple ID file: condition_drug_seq-CKD.id
        return ret['id']
    @staticmethod
    def getCSVFile(cohort, **kargs): 
        ret = setName(cohort, **kargs)  # exmaple CSV file: condition_drug_timed_seq-CKD.csv
        return ret['csv']
    @staticmethod
    def getDocFile(cohort, **kargs): 
        ret = setName(cohort, **kargs)  # exmaple document file: condition_drug_seq-CKD.dat
        return ret['dat']
    
    @staticmethod
    def verifyIDs(person_ids):  # [protocol]
        def has_dup(alist): 
            return len(alist) != len(set(alist))
        idx = person_ids

        # IDs must be in ascending order (for consistency); data labeling may depend on this ordering
        assert all(idx[i] <= idx[i+1] for i in xrange(len(idx)-1)), "IDs are not in ascending order!"
        assert not has_dup(idx), "IDs contain duplicates!"

        return

    @staticmethod
    def doctypeToStem(doctype='doc'):
        """
        Naming of the coding sequence file depends on the doctype (which provides clues for the sequence content)

        Related
        -------
        seqReader.makeLabeledSeqFile() <- TDoc  
        """
        if doctype is None: doctype = 'doc'
        stem = 'seq'  # doctype == 'doc'
        # if doctype.startswith('doc'):
        #     stem = 'seq' 
        if doctype.startswith('time'): # timed: sequences + timestamps
            stem = 'timed_seq' 
        elif doctype.startswith('label'): # labeled: sequences + timestamps + labels
            stem = 'labeled_seq' 
        elif doctype.startswith('vi'): # visit: sequences in units of visits (obsolete)
            stem = 'visit_seq' 
        elif doctype.startswith('me'): # meta: sequences + timestamps + labels + all kinds of labels 
            stem = 'meta_seq'
        else: 
            # use default 
            stem = 'seq'
            assert doctype == 'doc', "Unknown document type: %s" % doctype

        return stem

    @staticmethod
    def getMCSName(**kargs):
        """
        Get the file name of the derived MCS file (transformed documents), which is created in parallel to 
        the training set created by document embedding. This file follows training set naming convention 
        instead of the source document convention as in {getName, getNameByContent}

        """ 
        raise NotImplementedError, "See tdoc.TDoc."

    @staticmethod
    def getNameByContent(cohort=None, doctype=None, seq_ptype='regular', ext='dat', doc_basename=None, meta=''):
        """
        Specificalized temporal documents that consist of only particular code types (e.g. diagnostic codes only)

        Memo
        ----
        1. see seqTransform 
           each transformation by default should also generate a document source. 
        """
        if doc_basename is None: doc_basename = TDoc.doc_basename # 'condition_drug' by default based on sequence content
        ptype = seq_ptype
        if ptype.startswith('d'): 
            doc_basename = 'condition'
        elif ptype.startswith('m'): 
            doc_basename = 'drug'
        elif ptype.startswith('l'): 
            doc_basename = 'lab'
        return TDoc.getName(cohort=cohort, doctype=doctype, doc_basename=doc_basename, ext=ext, meta=meta)

    @staticmethod
    def getName(cohort=None, doctype=None, doc_basename=None, ext='dat', meta=''):  # ext: {'dat', 'csv', }
        """

        Memo
        ----
        1. examples 
          
           labeled document source 
                condition_drug_labeled_seq-CKD.csv

                Usually created via processDocuments.make_labeled_docs() 

           augmented/unlabeled document source with timestamps 
                condition_drug_timed_seq-augmented-CKD.csv

        """
        if doctype is None: doctype = TDoc.read_mode
        if doc_basename is None: doc_basename = TDoc.doc_basename # based on sequence content

        # seq_ptype = normalize_ptype(seq_ptype)
        cohort_name = cohort

        # ext = 'dat'   # may be changed to .csv file with header ['sequence', 'timestamp', 'label', ]
        # ifile = '%s_seq-%s.%s' % (doc_basename, cohort_name, ext) if cohort_name is not None else '%s_seq.%s' % (doc_basename, ext)
        stem = TDoc.doctypeToStem(doctype)

        # e.g. condition_drug_labeled_seq-CKD.csv
        # doc_basename: sequence content <- getNameByContent(cohort=None, doctype=None, seq_ptype='regular', ext='dat', doc_basename=None)
        # stem: document type (timed, labeled, meta, visit)
        # cohort
        # ext

        # e.g. condition_drug_timed_seq-augmented-CKD.csv
        if meta: stem = '%s-%s' % (stem, meta) # use meta for secondary ID (e.g. augmented/unlabeled document source(s))
        if cohort_name is not None: stem = '%s-%s' % (stem, cohort_name)

        ifile = '%s_%s.%s' % (doc_basename, stem, ext)

        # if doctype == 'timed': # 
        #     ifile = '%s_timed_seq-%s.%s' % (doc_basename, cohort_name, ext) if cohort_name is not None else '%s_timed_seq.%s' % (doc_basename, ext)
        # elif doctype == 'labeled': # labeled data, deidentified 
        #     ifile = '%s_labeled_seq-%s.%s' % (doc_basename, cohort_name, ext) if cohort_name is not None else '%s_labeled_seq.%s' % (doc_basename, ext)
        # elif doctype.startswith('v'): # visit
        #     ifile = '%s_visit_seq-%s.%s' % (doc_basename, cohort_name, ext) if cohort_name is not None else '%s_visit_seq.%s' % (doc_basename, ext)
        # else: 
        #     assert doctype == 'doc', "Unknown document type: %s" % doctype
            # raise ValueError, "Unknown document type: %s" % doctype
        # [todo] also consider specialized files (e.g. -diag, which contains only diagnostic codes)
        return ifile

    @staticmethod
    def getPathsByCohort(cohort, prefix=None, meta=''):
        assert cohort is not None 
        return TDoc.getPaths(cohort, basedir=prefix, ifiles=[], meta=meta) 

    @staticmethod
    def getPathsByTypes(cohort, prefix=None, doctypes=None, ext='csv', meta=''): 
        """
        Sometimes, we don't know if the source document has labels or not; may want 
        to try doctype=labeled followed by doc_type='timed' 

        """
        if doctypes is None: doctypes = ['labeled', 'timed', 'doc', ]
        if not hasattr(doctypes, '__iter__'): doctypes = [doctypes, ]
        
        fpaths = []
        for doctype in doctypes: 
            fpaths.extend(TDoc.getPaths(cohort, basedir=prefix, doctype=doctype, ifiles=[], ext=ext, meta=meta))
            if fpaths: break 
        return fpaths

    @staticmethod
    def getAugmentedDocPaths(cohort, prefix=None, doctype='timed', ifiles=[], ext='csv', verify_=False, meta=None): 
        # [note] prefix ~ basedir in getPaths

        # some preliminary rule for user-defined ID: avoid dup
        meta_base = 'augmented'
        if meta: 
            meta = str(meta)
            if meta.startswith('augment'): 
                meta  = meta_base
            else: 
                meta = '%s-%s' % (meta_base, meta)
        else: 
            meta = meta_base
        
        if prefix is None: prefix = seqparams.getCohortGlobalDir(cohort)  # this is where augmented data goes by default
        assert os.path.exists(prefix), "Invalid basedir: %s" % prefix
        return TDoc.getPaths(cohort, basedir=prefix, doctype=doctype, ifiles=ifiles, ext=ext, meta=meta, verify_=verify_)

    @staticmethod
    def getPaths(cohort=None, **kargs): # [params] doctype=None, doc_basename=None, ext='dat', basedir=None, ifiles=None
        """
        Formulate paths to sequencing source files. 
        
        Params
        ------
        basedir
        ifiles: files or paths to files

        verify_: if True, check if each file path is valid (existency)

        Use 
        ---
        Input: cohort, doctype, seq_ptype, doc_basename, ext + basedir
        Output: a list of paths to the docuemnt source file  
                cohort-specific usually has only one source 
                => use getPath()

        
        Memo
        ----
        1. example path (cohort='PTSD' doctype='doc')
           <projdir>/tpheno/data-exp/condition_drug_seq-PTSD.dat
        """
        basedir = kargs.get('basedir', None)
        if basedir is None: basedir = TDoc.prefix # sys_config.read('DataExpRoot') # getCohortGlobalDir(cohort) | basedir, create_dir
        assert os.path.exists(basedir), "Invalid document source directory: %s" % basedir

        tVerify = kargs.get('verify_', False)

        # [params] input(s) 
        #          use ifile with only 1 input, use ifiles with multiple input files
        ifiles = kargs.get('ifiles', [])  # don't use None as default
        assert hasattr(ifiles, '__iter__'), "Invalid input format: %s" % ifiles
        if kargs.has_key('ifile'):  # not preferable, use ifiles instead
            ifile = kargs['ifile']
            assert isinstance(ifile, str)
            ifiles.append(ifile)
            ifiles = list(set(ifiles)) # remove dups

        # no input given, use cohort-specific default
        if not ifiles: 
            # kargs.pop('cohort')

            assert cohort is not None, "Please specify a cohort (e.g. PTSD)"
            fp0 = os.path.join(basedir, TDoc.getNameByContent(cohort=cohort, doctype=kargs.get('doctype', TDoc.read_mode), 
                                                        seq_ptype=kargs.get('seq_ptype', 'regular'),  # determines the prefix (e.g. condition_drug, condition)
                                                        doc_basename=kargs.get('doc_basename', None),  # None: lazy default
                                                        ext=kargs.get('ext', 'dat'), meta=kargs.get('meta', '')))
            ifiles = [fp0, ]

        # normalize, ensure that each file has a full path
        for i, f in enumerate(ifiles): 
            # contain rootdir? 
            rootdir, fname0 = os.path.dirname(f), os.path.basename(f) 
            if not rootdir: 
                if i == 0: print('getPaths: use default basedir: %s' % basedir)
                assert os.path.exists(basedir), "Invalid base directory: %s" % basedir
                rootdir = basedir
            fp = os.path.join(rootdir, fname0)
            if tVerify: assert os.path.exists(fp), "Invalid input source file: %s" % fp
            ifiles[i] = fp
        return ifiles 
    @staticmethod
    def getPath(cohort=None, **kargs): 
        fpaths = TDoc.getPaths(cohort=cohort, **kargs)
        assert len(fpaths) > 0, "No input document found with cohort: %s" % cohort
        return fpaths[0]

    @staticmethod
    def verifyPaths(ifiles):
        # if not ifiles: 
        #     ifiles = TDoc.getPaths()
        for i, fp in enumerate(ifiles): 
            if not os.path.exists(fp): 
                raise RuntimeError, "Invalid input source file: %s" % fp

    # [formatting]
    @staticmethod
    def strToSeq(df, col='sequence'): # [todo] consider visits
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


### end class TDoc 

class TSet(object):
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

    Related 
    ------- 
    vector.D2V 
        model directory and training set directory share the same prefix directory (same parent)
        see seqmaker.seqClassify.makeTSetCombined2() for an example  ... 1.20.18

    """
    index_field = 'index'
    date_field = 'date'
    target_field = 'target'  # usually surrogate labels
    target_type = 'train'
    annotated_field = 'annotated'
    content_field = 'content'  # representative sequence elements 
    label_field = 'mlabels'  # multiple label repr of the underlying sequence (e.g. most frequent codes as constituent labels)

    meta_fields = [target_field, target_type, content_field, label_field, index_field, date_field, annotated_field, ]
    # feature_fields = [] # use getFeatureColums

    seq_ptype = 'regular' # sequence content type for training data
    sep = ','  # attribute/column separater

    generic_feature_prefix = 'f'
    dir_train = 'train'
    dir_test = 'test'
    dir_dev = 'dev'  # for model selection, hyperparameter tuning

    # specialized training set
    # label types defined in TDoc: 
    #    fLabels = ['label', 'label_lcs', 'label_unigram', 'label_bigram', 'label_trigram', 'label_4gram', ] 
    labelID_lcs = 'Llcs'

    @staticmethod
    def getName(cohort=None, d2v_method='d2v', **kargs): 
        """

        Params
        ------
        a. cohort
        b. identifier 
           seq_ptype, d2v_method, (w2v_method)

           suffix 

        c. vaildation index 
           index 
        """
        # training set names 
        stem = kargs.get('file_stem', 'tset')
        identifier = 'default'
        if cohort is None: 
            # cohort = System.cohort  # group ID
            cohort = 'generic'
            print('TSet> Warning: No cohort given > used system default: %s' % cohort)

        seq_ptype = kargs.get('seq_ptype', 'regular') # default, use entire coding systems to train document vectors
        w2v_method = kargs.get('w2v_method', None) # default, not included in the naming
        if w2v_method is not None: 
            identifier = '%s-%s-%s' % (seq_ptype, w2v_method, d2v_method)  # no need to include w2v_method (which is subsumed by d2v_method)
        else: 
            identifier = '%s-%s' % (seq_ptype, d2v_method)

        # secondary ID: 
        # 1. may want to distinguish training set with different coding content, leave this to suffix 
        # 2. training set derived from including augmented/unlabeled data
        suffix = kargs.get('suffix', None)
        if suffix is not None: 
            identifier = '%s-%s' % (identifier, suffix) 

        print('TSet> Params: cohort: %s, file id: %s' % (cohort, identifier))

        index = kargs.get('index', None)  # cross validation, multile training and test sets
        extension = kargs.get('ext', 'csv')   # .npz for sparse matrix
        if index is None: # ignore index 
            fname = '%s-ID%s-G%s.%s' % (stem, identifier, cohort, extension)  
        else: 
            fname = '%s-n%s-ID%s-G%s.%s' % (stem, index, identifier, cohort, extension)

        return fname

    @staticmethod
    def getDocName(cohort=None, d2v_method='d2v', **kargs): # MDS parallel to tset 
        """
        Parallel to getName(), this method is used to name the byproduct MCS associated with the training set (generated via 
        document embedding methods). 
        """
        kargs['file_stem'] = 'mcs'
        return TSet.getName(cohort=cohort, d2v_method=d2v_method, **kargs)

    @staticmethod
    def getNameLCS(cohort=None, d2v_method='d2v', **kargs): 
        kargs['suffix'] = 'Llcs'  # labeling scheme
        return TSet.getName(cohort=cohort, d2v_method=d2v_method, **kargs)

    @staticmethod
    def getWorkDir(**kargs):
        return TSet.getPath(**kargs) 
    @staticmethod
    def getPath(cohort=None, dir_type='train', create_dir=True):
        if cohort in (None, '?', 'n/a', ): 
            cohort = System.cohort  # group ID
            print('TSet> Warning: No cohort given > used system default: %s' % cohort)

        # structure: ./data/<cohort>/{train, test}
        rootdir = getCohortLocalDir(cohort=cohort)  # in global utility
        if dir_type is None:  # e.g. training data for cluster analysis may not need to distinguish training and test directories
            return rootdir 

        assert os.path.exists(rootdir)
        fpath = os.path.join(rootdir, dir_type)
        if not os.path.exists(fpath) and create_dir: 
            print('TSet> Create a new directory (direcotry type=%s): %s' % (dir_type, fpath))
            os.makedirs(fpath) 
        return fpath

    @staticmethod
    def getHeader(X):
        f_prefix = TSet.generic_feature_prefix
        return ['%s%s' % (f_prefix, i) for i in range(X.shape[1])]
    @staticmethod
    def getFeatureColumns(n):  # training set dataframe -> (X, y) format; this avoids knowing all meta fields which may expand dynamically
        f_prefix = TSet.generic_feature_prefix
        return ['%s%s' % (f_prefix, i) for i in range(n)]     

### end class TSet  (also see tset module for child classes)

class Pathway(object): 
    sep = '|'  # separater in pathway dataframe

    # n_uniq: a diversity measure; number of unique tokens within a given LCS
    header_global_lcs = ['lcs', 'length', 'count', 'n_uniq', 'df']  # n_docs, df: document frequency

    # LCS 
    lcs_nomatch = 'No_Match'
    lcs_sep = ' '  # the LCS string in the dataframe uses this to separate codes
    scope = ['global', 'local', ]  # local includes LCSs derived from clustering 
    lcs_policy = ['freq', 'diversity', 'length', 'df', ] # freq: frequency
    ng_policy = ['df', 'tf-idf']

    @staticmethod
    def getName(scope='global', policy='freq', suffix='default', file_stem='pathway'): # pairing_policy='rand'
        # e.g. pathway_global-freq-regular-iso-LCKDStage1.csv
        print('Pathway.getName: scope: %s, policy: %s, suffix: %s' % (scope, policy, suffix))
        identifier = '%s-%s-%s' % (scope, policy, suffix)
        fname_out = '%s_%s.csv' % (file_stem, identifier) # identifier: (tset_type, cluster_method, seq_ptype, d2v_method)
        return fname_out
    @staticmethod
    def getPathwayName(**kargs):  # pathway file name
        kargs['file_stem'] = 'pathway'
        return Pathway.getName(**kargs) 
    @staticmethod
    def getNGramName(**kargs):  # ngram file name
        kargs['file_stem'] = 'ngram'
        return Pathway.getName(**kargs) 
    @staticmethod
    def getWorkDir(cohort, dir_type='pathway', create_dir=True):
        if cohort is None: 
            cohort = System.cohort  # group ID
            print('Pathway> Warning: No cohort given > used system default: %s' % cohort)

        # structure: ./tpheno/seqmaker/data/<cohort>/pathway
        rootdir = getCohortLocalDir(cohort=cohort)  # in global utility
        if dir_type is None:  
            return rootdir 

        assert os.path.exists(rootdir), "Pathway.getWordDir> Invalid path: %s" % rootdir
        fpath = os.path.join(rootdir, dir_type)
        if not os.path.exists(fpath) and create_dir: 
            print('Pathway> Create a new directory (direcotry type=%s): %s' % (dir_type, fpath))
            os.makedirs(fpath) 
        return fpath
    @staticmethod
    def getPath(**kargs):
        return Pathway.getWorkDir(**kargs)
    @staticmethod
    def getFullPath(cohort, dir_type='pathway', **kargs): 
        """

        Memo
        ----
        1. example path to a LCS-pattern file
           <prefix>/tpheno/seqmaker/data/CKD/pathway/lcs_local-df-regular-iso-LControl-Uenriched.csv
               
                file_stem: lcs 
                scope: local 
                policy: df
                suffix: regular-iso-L<label> 
                        where label <- Control 
                meta: a pattern type {'enriched', 'rare', }

        """
        import os 

        # file parameters
        pattern_type = kargs.get('pattern_type', 'lcs')  # {'lcs', 'ngram', 'pathway', }
        scope = kargs.get('scope', 'global')
        policy = kargs.get('policy', 'freq')
        # pairing_policy = kargs.get('pairing_policy', 'rand')
        suffix = kargs.get('suffix','default')
        fname = Pathway.getName(scope=scope, policy=policy, suffix=suffix, file_stem=pattern_type) 

        # dir parameters: cohort, dir_type
        workdir = Pathway.getWorkDir(cohort=cohort, dir_type=dir_type, create_dir=True)
        return os.path.join(workdir, fname)

    @staticmethod
    def do_slice(**kargs): # only for file ID, see secondary_id()
        if not kargs.has_key('slice_policy'): return False
        if kargs['slice_policy'].startswith(('noop', 'reg', 'compl', )):  # effective operations 
            return False 
        return True
    @staticmethod
    def file_id(**kargs): # params: {cohort, scope, policy, seq_ptype, slice_policy, consolidate_lcs, length}
        # [note] policy: 'freq' vs 'length', 'diversity'
        #        suppose that we settle on 'freq' (because it's more useful) 
        #        use pairing policy for the policy parameter: {'random', 'longest_first', }
        adict['cohort'] = kargs.get('cohort', 'generic')  # this is a mandatory arg in makeLCSTSet()
        adict['scope'] = ltype = kargs.get('scope', 'global')  # global: Common LCSs defined in global scope (as opposed to cluster or local scopes)
        adict['policy'] = kargs.get('policy', 'freq') # definition of common LCSs

        # use {seq_ptype, slice_policy, length,} as secondary id
        adict['suffix'] = adict['meta'] = suffix = Pathway.secondary_id(**kargs)
        return adict
    @staticmethod
    def secondary_id(**kargs): # attach extra info in suffix
        suffix = kargs.get('suffix', kargs.get('seq_ptype', 'regular')) # vector.D2V.d2v_method
        if Pathway.do_slice(): suffix = '%s-%s' % (suffix, kargs['slice_policy'])
        if kargs.get('consolidate_lcs', True): suffix = '%s-%s' % (suffix, 'iso') # LCSs up to permutations consider as the same? 
        suffix = '%s-len%s' % (suffix, kargs.get('min_length', 3))
        return suffix 

    @staticmethod
    def load(cohort, dir_type='pathway', **kargs): # [params] scope='global', policy='freq', suffix='default',
        # from pandas import DataFrame
        import pandas as pd

        # file parameters
        # [todo] class parameters?
        pattern_type = kargs.get('pattern_type', 'lcs')  # {'lcs', 'ngram', 'pathway', }
        # ctype = kargs.get('seq_ptype', 'regular')
        scope = kargs.get('scope', 'global')
        policy = kargs.get('policy', 'df')
        suffix = kargs.get('suffix','default')

        # [note] suffix can be white spaces
        suffix = suffix.replace(" ", "")
        fpath = Pathway.getFullPath(cohort=cohort, 
                    scope=scope, policy=policy, suffix=suffix, dir_type=dir_type, pattern_type=pattern_type) # [params] dir_type='pathway'
        # assert os.path.exists(fpath), "Invalid input path: %s" % fpath 
        df = None
        if os.path.exists(fpath): 
            df = pd.read_csv(fpath, sep=Pathway.sep, header=0, index_col=False, error_bad_lines=True)
            print('Pathway> loaded %s data from:\n%s\n' % (pattern_type, fpath))
        else: 
            # e.g. /data/<cohort>/pathway/pathway_global-freq-CTyperegular.csv
            print "Pathway.load> Warning: Non-existent input path: %s" % fpath 

        return df # [output] a dataframe or None
    @staticmethod
    def save(df, cohort, dir_type='pathway', **kargs): 
        import pandas as pd

        # assert set(Pathway.header_global_lcs).issubset(df.columns), \
        #     "Not all attributes are included in the input data (cols: %s)" % df.columns.values

        # file parameters
        pattern_type = kargs.get('pattern_type', 'lcs')  # {'lcs', 'ngram', 'pathway', }
        scope = kargs.get('scope', 'global')
        policy = kargs.get('policy', 'freq')
        suffix = kargs.get('suffix','default')

        # [note] suffix can be white spaces
        suffix = suffix.replace(" ", "")
        fpath = Pathway.getFullPath(cohort=cohort, 
                    scope=scope, policy=policy, suffix=suffix, dir_type=dir_type, pattern_type=pattern_type) # [params] dir_type='pathway'
        # condition: lcsmap has been computed
        df.to_csv(fpath, sep=Pathway.sep, index=False, header=True) 
        print('Pathway> saved %s data to:\n%s\n' % (pattern_type, fpath))

        return

### enc class Pathway

class ClassifierParams(object): 
    training_set_file = 'tset'
    roc_plot = 'roc'
    pca_plot = 'pca'

    c_scope = np.logspace(-3, 3, 7)
    c_min, c_max = (c_scope[0], c_scope[-1]*100)
    penalty = ['l1', 'l2']

    params = {} 
   
    # SVC(kernel='linear', C=10)
    params['linear_svm'] = {'kernel': 'linear', 'C': 1, }
    # SVC(kernel='rbf', C=100, gamma=0.001)
    params['rbf_svm'] = {'kernel': 'rbf', 'C':1, 'gamma': 0.001}

    # alpha = 0.1; lasso = Lasso(alpha=alpha)
    params['lasso'] = {'alpha': 0.1, }

    # ElasticNet(alpha=alpha, l1_ratio=0.7)
    params['elasticnet'] = {'alpha': 0.1, 'l1_ratio':0.7}   # l1_ration -> 1, more L1, less L2

    # used in feature selection
    multirun_cycle = 10
    best_ensemble_clf = False  # if True: given a set of h2s (high-level classifier), choose the best one
    n_jobs = -1  # use all CPUs 

    is_dir_configured = False 
    delay = 5

    # sample weight 
    weight_sample = False

    @staticmethod
    def getParams(clf, penalty=None, **kargs): 
        name = clf.__class__.__name__
        c_scope = kargs.get('c_scope', ClassifierParams.c_scope)
        if isinstance(clf, LogisticRegression) or name.startswith('logist'):
            if penalty is None: penalty = ClassifierParams.penalty
            if isinstance(penalty, str): 
                assert penalty in ('l1', 'l2', )  # elasticnet? 
                penalty = [penalty]
            return [{'C': c_scope, 'penalty': penalty}]
        elif isinstance(clf, LassoLogistic) or name.startswith('lasso'):
            return [{'alpha': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0], }]
        elif isinstance(clf, SVC) or name.lower() in ('svm', 'svc', ):
            # rep: list of dictionary of specs 
            return [{'kernel': ['rbf', ], 'gamma': np.logspace(-4, 0, 5),
                       'C': c_scope},
                      {'kernel': ['linear'], 'C': c_scope}, 
                      {'kernel': ['poly'], 'degree': [2, 3], 'C': c_scope}, 
                    ]
        return [{'C': c_scope}]

    @staticmethod
    def getClassifier(name='logistic', penalty=None, **kargs):
        """

        (*) LogisticRegression
            + http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
              C: the smaller, the stronger the regularization

        Note
        ----
        1. The loss function to be used. 
           Defaults to 'hinge', which gives a linear SVM. 
           The 'log' loss gives logistic regression, a probabilistic classifier. 
           'modified_huber' is another smooth loss that brings tolerance to outliers as well as probability estimates. 
           'squared_hinge' is like hinge but is quadratically penalized. 
           'perceptron' is the linear loss used by the perceptron algorithm. 
           The other losses are designed for regression but can be useful in classification as well. 
        """
        reg = kargs.get('reg', None)  # regularization strength
        if not kargs.has_key('class_weight'): kargs['class_weight']='balanced'

        # if c_scope is None: c_scope = Params.c_scope
        if name.startswith('logis') or isinstance(name, LogisticRegression): 
            if penalty is None: penalty = ['l1', 'l2']
            if isinstance(penalty, str): 
                assert penalty in ('l1', 'l2', )  # elasticnet? 
                penalty = [penalty]
            if reg is None: reg = ClassifierParams.c_scope
            params = [{'C': reg, 'penalty': penalty }] # np.logspace(-1, 2, 4)
            
            # adjust weights inversely proportional to class frequencies in the input data
            # cw = kargs.get('class_weight', 'balanced')

            if len(penalty) == 1: 
                clf = LogisticRegression(tol=0.01, penalty=penalty[0], **kargs)
            else: 
                clf = LogisticRegression(tol=0.01, **kargs) # set other params later

        elif name in ('svm', 'svc', ) or isinstance(name, SVC):
            if not kargs.has_key('probability'): kargs['probability']=True
            clf = SVC(**kargs)
            if reg is None: reg = ClassifierParams.c_scope
            params = ClassifierParams.getParams(clf=clf)
            # params = [{'kernel': ['rbf'], 'gamma': np.logspace(-4, 0, 5),
            #            'C': reg},
            #           {'kernel': ['linear'], 'C': reg}]
        elif name.lower() in ('sgd', ): # stochastic gradient descent 
            # l1_ratio = 0.15 by default (if 1 => l1 penalty, lasso)
            # [1]
            cw = kargs.get('class_weight', 'balanced')
            if len(penalty) == 1: 
                clf = SGDClassifier(class_weight=cw, penalty=penalty)  
            else: 
                clf = SGDClassifier(class_weight=cw, penalty='elasticnet')  # 'elasticnet' be default, designed mainly for regression
              
        else:
            raise ValueError, "learnerConfig.getClassifier> Use inherited methods for %s" % name 

        return (clf, params)

### end class ClassifierParams

class Graphic(object): 
    
    supported_formats = ['eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff', ]

    @staticmethod
    def getName(cohort='generic', **kargs):  
        """

        Params
        ------
        identifier
           seq_ptype, w2v_method, d2v_method: checked only if identifier not given
        ext_plot 
        
        prefix: central description of this plot 
        
        Optional 
        --------
        seq_ptype, d2v_method, w2v_method
        index: kth fold of cross valiation 
        
        """
        # training set names 
        identifier = kargs.get('identifier', '')
        if cohort is None: 
            # cohort = System.cohort  # group ID
            cohort = 'generic'
            print('TSet> Warning: No cohort given > used system default: %s' % cohort)

        seq_ptype = kargs.get('seq_ptype', 'regular') # default, use entire coding systems to train document vectors
        d2v_method = kargs.get('d2v_method', 'pv-dm2')  # pv-dm2: pv-dm + pv-dbow
        w2v_method = kargs.get('w2v_method', None) # default, not included in the naming
   
        # [params]
        # identifier = '' 
        # for param in [seq_ptype, d2v_method, w2v_method, ]: 
        #     identifier += 
        if not identifier: 
            if w2v_method is not None: 
                identifier = '%s-%s-%s' % (seq_ptype, w2v_method, d2v_method)  # no need to include w2v_method (which is subsumed by d2v_method)
            else: 
                identifier = '%s-%s' % (seq_ptype, d2v_method)

        # secondary ID: may want to distinguish training set with different coding content, leave this to suffix 

        print('Graphic> Params: cohort: %s, file id: %s' % (cohort, identifier))

        ext_plot = kargs.get('ext_plot', 'tif')
        assert ext_plot in Graphic.supported_formats

        prefix = kargs.get('prefix', 'plot')   # graphic description (e.g. ROC)
        index = kargs.get('index', None)  # cross validation, multile training and test sets
        if index in (None, 'n/a'): 
            fname = '%s-ID%s-G%s.%s' % (prefix, identifier, cohort, ext_plot)
        else: 
            fname = '%s-n%s-ID%s-G%s.%s' % (prefix, index, identifier, cohort, ext_plot)

        return fname

    @staticmethod
    def getPath(cohort=None, dir_type='plot', create_dir=True):
        if cohort is None: 
            cohort = 'generic' # System.cohort  # group ID
            print('TSet> Warning: No cohort given > use default: %s' % cohort)

        # structure: ./data/<cohort>/<dir_type>  | ./data/<cohort>  if dir_type <- None
        rootdir = getCohortLocalDir(cohort=cohort)  # in global utility
        if dir_type is None:  # e.g. training data for cluster analysis may not need to distinguish training and test directories
            return rootdir 

        assert os.path.exists(rootdir)
        fpath = os.path.join(rootdir, dir_type)
        if not os.path.exists(fpath) and create_dir: 
            print('Graphic> Create a new directory (direcotry type=%s): %s' % (dir_type, fpath))
            os.makedirs(fpath) 
        return fpath
    @staticmethod
    def getFullPath(cohort=None, **kargs): 
        pass

### end class Graphic


def summary(): 
    pass

def makeID(params, sep='-', null_chars=None):
    def is_null(p): 
        if p is None: return True 
        if len(p) == 0: return True
        if p.lower() in null_chars: return True
        return False

    if isinstance(params, str): 
        params = [params, ]
    if not params: 
        params = ['test', ]
    assert hasattr(params, "__iter__")

    if null_chars is None: null_chars = ['n/a', '?', 'na', 'null', '', ]
    
    eff_params = []
    for param in params: 
        if not is_null(param):  # not None, ''
            eff_params.append(param)

    return sep.join(eff_params)

def get_testdir(prefix=None, cohort=None, topdir='test'): 
    """
    Get module-wise test directory, which is located at <module>/<testdir> 
    where <module> in this case is 'seqmaker'

    Related
    -------
    1. generic version: batchpheno/futils.py  (i.e. file utility)
    """
    if prefix is None: prefix = os.getcwd()
    testdir = os.path.join(prefix, topdir)
    if not os.path.exists(testdir): os.makedirs(testdir) # test directory
    return testdir

def get_basedir(prefix=None, cohort=None, topdir='data'): 
    if prefix is None: prefix = os.getcwd()
    if cohort is not None: 
        topdir = os.path.join(topdir, cohort)  # <topdir>/<cohort> on unix
    basedir = os.path.join(prefix, topdir) # [I/O] sys_config.read('DataExpRoot')
    if not os.path.exists(basedir): os.makedirs(basedir) # base directory
    return basedir

def get_ifile(composition='condition_drug', cohort='diabetes'):
    cohort_name = cohort

    # [todo] use the pattern: seq_%s-%s.dat 
    return '%s_seq-%s.dat' % (composition, cohort_name) if cohort_name is not None else '%s_seq.dat' % cohort_name

def arg(names, default=None, **kargs):  # [refactor] utils.arg
    val, has_value = default, False
    if hasattr(names, '__iter__'): 
        for name in names: 
            try: 
                val = kargs[name]
                has_value = True 
                break
            except: 
                pass 
    else: 
        try: 
            val = kargs[names]
            has_value = True
        except: 
            print('warning> Invalid key value: %s' % str(names))

    if not has_value:    
        print('warning> None of the keys were given: %s' % names) 
    return val

def normalize_ttype(n_classes):  # standardize tset_type or training set type 
    tset_type = 'unknown'
    if n_classes == 1: 
        tset_type = 'unary' # one class, binary, multiclass
    elif n_classes == 2: 
        tset_type = 'binary'
    else: 
        tset_type = 'multiclass' 
    return tset_type 

def normalize_ptype(seq_ptype, **kargs):
    print('seqparams> Warning: use normalize_ctype instead. ptype could refer to policy type in seqCluster.')
    return normalize_ctype(seq_ptype, **kargs)

def normalizeSeqContentType(seq_ptype, **kargs): 
    return normalize_ptype(seq_ptype, **kargs)
def normalize_ctype(seq_ptype, **kargs): 
    """
    Normalize sequence content type (e.g. diag for diagnostic-code-only sequences)

    Sequence content types: 
    'regular'
    'random'
    'overlap_ngram' where 'n' depends on input 'ngram'
    'diag'
    'med'
    'lab'
    """

    # ngram = kargs.get('ngram', 3) # the n in n_gram overlap 
    ptype = seq_ptype  
 
    if ptype is None or ptype.startswith(('reg', 'mix', 'noop', 'full', 'compl')): 
        # reg: {regular, full, complete} sequences
        # mix: mixture of all types of codes
        # compl: {regular, full, complete} sequences
        # noop: no cutting (as opposed to pre-diagnostic, post-diagnostic sequences)
        # full: {regular, full, complete} sequences
        ptype = 'regular'
    elif ptype.startswith('rand'): 
        ptype = 'random'
    elif ptype.startswith('overlap'):
        ngram = kargs.get('ngram', 3)
        ptype = 'overlap_%dgram' % ngram  # ABCD(3) => ABC, BCD, 
    elif ptype.startswith('d'): 
        ptype = 'diag'
    elif ptype.startswith('m'): 
        ptype = 'med'
    elif ptype.startswith('l'): 
        ptype = 'lab'
    # elif ptype.startswith(('pri', 'pre')):
    #     ptype = 'prior'  # pre-diagnostic 
    # elif ptype.startswith('post'):
    #     ptype = 'posterior'  
    else: 
        print("warnig> unknown sequence pattern type: %s => set to 'regular'" % ptype)
        ptype = 'regular'

    # coding lexicon policy 
    simplified = kargs.get('simplify_code', False)   
    if simplified: 
        ptype = '%s_%s' % ('simplified', ptype)

    return ptype

def normalize_lcs_selection(opt):
    # options: frequency/local (f), global frequency (g), length (l), diversity (d)
    if opt.startswith( ('f', 'loc')): 
        opt = 'freq'
    elif opt.startswith('g'):
        opt = 'global_freq'
    elif opt.startswith('len'): 
        opt = 'length'
    elif opt.startswith('div'): 
        opt = 'diversity'
    else: 
        raise ValueError, "Unknown LCS selection policy: %s" % opt
    return opt

def normalize_w2v_method(w2v_method, **kargs):
    method = 'sg'
    if isinstance(w2v_method, int): # 0: cbow, 1: skip-gram
        if w2v_method == 0: 
            method = 'cbow'
        elif w2v_method == 1: 
            method = 'sg'
        else: 
            print('warning> unrecognized input %d => use default %s' % (w2v_method, method))
    elif isinstance(w2v_method, str): 
        method = w2v_method.lower()
        if w2v_method.startswith(('s', 'skip', )): 
            method = 'sg'
        elif w2v_method.startswith(('c', 'cb')):
            method = 'cbow'
        elif w2v_method.startswith(('pv-db', 'pvdb')):  # really is D2V 
            method = 'pv-dbow'
        elif w2v_method.startswith(('pvdm', 'pv-dm', )): 
            method = 'pv-dm'
        else: 
            print('info> advanced word embedding method? %s' % method)
            # noop 
    else: 
        raise ValueError, "Invalid w2v value: %s" % str(w2v_method)

    return method

def normalize_d2v_method(d2v_method): # [todo]
    method = 'tfidf'
    
    if d2v_method.startswith(('pv-db', 'pvdb')):  # really is D2V 
        method = 'pv-dbow'
    elif d2v_method.startswith(('pvdm', 'pv-dm', )): 
        method = 'pv-dm'    

    return method

def name_image_file(descriptor, **kargs):  # e.g. heatmap
    ext = kargs.get('graph_ext', 'tif')  

    n_sample = kargs.get('n_sample', kargs.get('n_points', None))
    seq_ptype = kargs.get('seq_ptype', 'regular')
    d2v_method = kargs.get('d2v_method', 'PVDM')
    # vis_method = kargs.get('descriptor', 'manifold') # visualization method
    identifier = '%s_%s_%s' % (descriptor, seq_ptype, d2v_method)

    if n_sample is not None: 
        return 'I%s-S%d.%s' % (identifier, n_sample, ext)
    return 'I%s.%s' % (identifier, ext)

def name_cluster_file(descriptor, **kargs):
    ext = kargs.get('graph_ext', 'tif')

    n_sample = kargs.get('n_sample', kargs.get('n_points', None))
    seq_ptype = kargs.get('seq_ptype', 'regular')
    d2v_method = kargs.get('d2v_method', 'PVDM')
    # vis_method = kargs.get('descriptor', 'manifold') # visualization method
    identifier = '%s_%s_%s' % (descriptor, seq_ptype, d2v_method)

    if n_sample is not None: 
        return '%s-S%d.%s' % (identifier, n_sample, ext)
    return '%s.%s' % (identifier, ext)

def getCohortSrcDir(cohort, basedir=None, create_dir=False):
    """
    Get the data source directory where external data, typically served as inputs, 
    are accessed. 

    This directory usually not the same as the output of the getCohortDir(), which 
    is used to access the data generated by this system (data-exp). 

    Note
    ----
    1. External data are those that are NOT generated by this system (i.e. tpheno modules)

    """
    if basedir is None: 
        # alternatively, os.path.join(os.getcwd(), 'data')  
        basedir = sys_config.read('DataIn') # <project>/data (data-in symblinked to data)   

    if cohort is None:  # None 
        fpath = basedir
    else: 
        fpath = os.path.join(basedir, cohort)

    if create_dir and not os.path.exists(fpath):
        print('getCohortSrcDir> creating new directory (cohort=%s): %s' % (cohort, fpath))
        os.makedirs(fpath) 

    return fpath

def getCohortGlobalDir(cohort, **kargs):
    return getCohortExpDir(cohort, **kargs)
def getCohortExpDir(cohort, basedir=None, create_dir=False):
    """

    Related 
    -------
    see getCohortDir()
    """
    assert cohort is not None
    if basedir is None: 
        basedir = sys_config.read('DataExpRoot')
    
    fpath = os.path.join(basedir, cohort)

    if create_dir and not os.path.exists(fpath):
        print('getCohortDir> creating new directory (cohort=%s): %s' % (cohort, fpath))
        os.makedirs(fpath) 

    return fpath 


# <basedir>/<cohort>/{train, test}
def getCohortLocalDir(cohort, **kargs):
    return getCohortDir(cohort, **kargs)
def getCohortDir(cohort, basedir=None, create_dir=False): 
    """
    Cohort-specific Diretctory (hosting outputs generated by the system)

    Related
    --------
    get_basedir()
    getCohortExpDir(): hosts cohort-specific outputs (e.g. d2v models) under data-exp (global scale)


    """ 
    assert cohort is not None

    if basedir is None: 
        basedir = os.path.join(os.getcwd(), 'data')  # or sys_config.read('DataExpRoot')
    
    fpath = os.path.join(basedir, cohort)
    if create_dir and not os.path.exists(fpath):
        print('getCohortDir> creating new directory (cohort=%s): %s' % (cohort, fpath))
        os.makedirs(fpath) 

    return fpath

# TDoc: friend functions
def getIDFile(cohort, **kargs):
    # [todo]
    ret = TDoc.setName(cohort, **kargs)  # exmaple ID file: condition_drug_seq-CKD.id
    return ret['id']
def getCSVFile(cohort, **kargs): 
    ret = TDoc.setName(cohort, **kargs)  # exmaple CSV file: condition_drug_timed_seq-CKD.csv
    return ret['csv']
def getDocFile(cohort, **kargs): 
    ret = TDoc.setName(cohort, **kargs)  # exmaple document file: condition_drug_seq-CKD.dat
    return ret['dat']

def t_path(**kargs): 

    ### paths: source files 
    fpaths = TDoc.getPaths(cohort='PTSD', doctype='doc', ifiles=kargs.get('ifiles', []), ext='csv', verify_=False) # doctype='timed'
    for i, fp in enumerate(fpaths): 
        print('[%d] %s' % (i, fp))

    return

def t_labeling(**kargs): 

    labels = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', 'CKD G1A1-control', 'Unknown', 
              'Control', 'CKD Stage 3', 'CKD Stage 3a', 'CKD Stage 3b', 'CKD Stage 2', 'CKD Stage 4', 
              'CKD Stage 5', 'CKD G1-control', 'ESRD on dialysis']

    lp = System.convert_labels(labels=labels, lmap={}, inplace=False)
    for i, label in enumerate(labels):
        print("[%d] %s -> %s" % (i+1, label, lp[i])) 

    return 

def test(**kargs):

    ### Get file path based on paramters
    # t_path(**kargs)

    ### Label mapping
    t_labeling(**kargs)

    return

if __name__ == "__main__": 
    test()




