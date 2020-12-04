import plotly.plotly as py
import plotly.graph_objs as go



# the smaller the size, the heigher the class weight
    class_weight_dict = eval_weights(y) # this has to come before binarzing
    print('... class weights:\n%s\n' % class_weight_dict)

    ### regularization 
    # early stopping
    spec.params_summary()
    target_metric = 'val_loss'  # note: monitor <- 'auc_roc' doesn't seem useful
    mode = eval_mode(target_metric)  # higher is better or lower is better? 

# binary encode labels
    lb = binarize_label(y_uniq)
    y_train = lb.transform(y_train)
    history = model.fit(X_train, y_train,
                  validation_split=spec.r_validation_split,  # e.g. 0.3
                  class_weight=class_weight_dict, # 
                  shuffle=spec.shuffle,
                  batch_size=spec.batch_size, epochs=spec.epochs, verbose=1,
                  callbacks=callback_list) 

    print('... history of metrics comprising: %s' % history.history.keys())
    print('... model fitting complete > starting model evaluation')



# [note] expensive! save intermmediate result if possible
        outputfile = 'analyze_strata.pkl'
        tAnalyzedMCS = False
        ret = {}
        try: 
            ret = pickle.load(open(os.path.join(os.getcwd(), outputfile), 'rb'))
        except: 
            print('analyze_strata> Could not find precomputed result from: %s ...' % outputfile)

        if not ret or len(ret['index']) == 0: 
            ret = analyzeMCS(Dr, Tr, lcs_set=lcs_candidates, make_color_time=False, check_df_min=False, 
                    save_result=True, outputfile=outputfile)  # index:lcsmap, frequency:lcsmapInvFreq


# small cohort 

def test(**kargs): 
    """

    Settings
    ---------
    seqClassify (base)
       use deep learning methods (e.g. LSTM) to classify concatenated visit vectors (instead of classical classifiers such as GBT)

       tSegmentVisit := True 
       tNNet := True

       drop_ctrl := False  ... retain control group

    Memo
    ----
    1. example configuration 
       nPerClass = None     # use all 
       nDPerClass = 3000

       max_visit_length = 50  # max number of tokens/codes in a visit segment
       n_timesteps = 10 
       
       => <prefix>/data/CKD/model/Pf100w5i50-regular-D3000-visit-M50t10.{dm, dbow}

    """
    def includeEndPoint(): 
        ret = {'meta': 'prior_inclusive', 'include_endpoint': True}
        return
    def is_default_visit(maxlen=100, nt=10, d2v='pv-dm2'): 
        if not d2v in ('pv-dm2', ): return False
        return maxlen == 100 and nt == 10
    def hasTSet(): 
        return X is not None and X.shape[0] > 0
    def doClassify(): 
        return tClassify and hasTSet()

    from pattern import ckd
    import timing  # track execution time

    # import seqConfig as sq

    cohort = 'CKD'
    policy_segment = 'regular'  # policy values: {'regular'/noop, 'two', 'prior', 'posterior', 'complete', 'visit', }
    
    # model visit-specific documents (in which each visit -> document vector i.e. visit vector)
    tNNet = True  # neural net mode

    # if tNNet is True, we probably want this to be True as well ...
    # With the classicial classifier, segmenting by visits is not recommended because there isn't a good mechanism for combining 
    # session vectors besides concatenating them, which leads to long, high-D vectors. 
    tSegmentVisit = True if tNNet else False 

    # predicate = ckd.isCase    # default predicate function for segmenting documents
    predicate = ckd.isCaseCCS if not policy_segment.startswith('reg') else None  # same as ckd.isCase plus minor error correction

    # C5K: 5K samples per class 
    # suffix_examples = {'C5K', }

    nPerClass = None # None: use all; 5000
    nDPerClass = None  
    d2v = None # None: default

    # meta serves as the file ID for d2v model, training set, etc. 
    # params: policy_segment, nPerClass + (max_visit_length, n_timesteps)

    n_features = 100 # vector.D2V.n_features * 2 if d2v in (None, 'pv-dm2', ) else vector.D2V.n_features
    n_timesteps = 20 # only keep track of the last k visits => n_timesteps: k
    
    # d2v model parameters
    window = 10
    n_iter = 20

    min_visit_length = window
    max_visit_length = 100    # default 100

    meta = '%s' % policy_segment  # tset size, include_endpoint? drop_nullcut?   e.g. C5K
    if nPerClass is not None: meta = '%s-C%d' % (meta, nPerClass) # max number of training instances per class
    if nDPerClass is not None: meta = '%s-D%d' % (meta, nDPerClass) # max number of documents per class

    meta_model = meta  # for model file, not need to train every time
    if tSegmentVisit: 
        if is_default_visit(maxlen=max_visit_length, nt=n_timesteps): # by default: max_visit_length=100, last_n_visits=10 
            meta = '%s-visit' % meta
        else: 
            # specify parameters 
            meta = '%s-visit-M%dt%d' % (meta, max_visit_length, n_timesteps) 
        
    include_endpoint, drop_nullcut, drop_ctrl = False, False, False  # only relevant when policy_segment is not 'regular'
    if policy_segment.startswith(('pri', 'post')): 
        drop_nullcut=True  # sequences where predicate does not apply (e.g. lack of decisive diagnosis) are nullcuts
        drop_ctrl=True   # need to drop control group because there may not exist data points for say in pre-diagnostic sequences

    ### document processing 
    
    # a. user-specified inputs 
    inputfile = None # 'condition_drug_labeled_seq-CKD.csv'
    inputdir = None # seqparams.getCohortGlobalDir(cohort='CKD0') # small cohort (n ~ 2.8K)
    secondary_id = 'dev'
    if inputfile is not None: 
        ipath = os.path.join(inputdir, inputfile)
        assert os.path.exists(ipath), "Invaid user input path:\n%s\n" % ipath
        meta_model = meta = '%s-%s' % (meta, secondary_id)

        n_features = 50
    # t_process_docs(cohort='CKD')

    ### configure system 
    sysConfig(cohort=cohort, d2v_method=d2v, seq_ptype='regular',
        n_features=n_features, window=window, n_iter=n_iter, 
        meta=meta, meta_model=meta_model)  # [params] d2v_method, seq_ptype

    ### Modeling starts from here ### 
    tMakeTset, tClassify = True, True 

    # a. full documents
    n_samples = 0; X = y = None
    if tMakeTset: 
        if tSegmentVisit: 
            X, y = t_model(min_ncodes=10, 
                        predicate=predicate, policy_segment=policy_segment, 

                        inputfile=inputfile, inputdir=inputdir, # user inputs

                        max_n_docs=nDPerClass,  # load only a max of this number of documents for each class
                        max_n_docs_policy='longest', 
                        
                        segment_by_visit=True, 
                        
                        # max_visit_length=max_visit_length, 
                        min_visit_length=min_visit_length, 
                        last_n_visits=n_timesteps,  # only keep track of the last 10 visits => n_timesteps: 10

                        load_model=True, test_model=False, 
                        include_augmented=False)
        else: 
            X, y = t_model(min_ncodes=10, 
                        predicate=predicate, policy_segment=policy_segment, 

                        inputfile=inputfile, inputdir=inputdir, # user inputs

                        segment_by_visit=False,
                        max_n_docs=nDPerClass, 
                        max_n_docs_policy='longest', 

                        load_model=True, test_model=False, 
                        include_augmented=False)  # use the default d2v method defined in vector module
        
        # n_features here is NOT necessarily identical to vector.D2V.n_features
        n_samples, n_features = X.shape[0], X.shape[1]
    
        # b. bag of words
        # t_model(min_ncodes=10, 
        #            load_model=False, test_model=False, 
        #            max_features=10000, 
        #            include_augmented=False)  # bag-of-words


        # c. pre-diagnostic, post-diagnostic segments 
        # [note] set max n_features to None to include ALL features
        # X, y = t_model(min_ncodes=10, 
        #                predicate=predicate, policy_segment=policy_segment, 

        #                # max_n_docs=None,  # 5000

        #                include_endpoint=include_endpoint,
        #                drop_nullcut=drop_nullcut,  
        #                load_model=False, test_model=False, 
        #                include_augmented=False)  # use the default d2v method defined in vector module

        # large CKD cohort | X (dim: (389350, 2000)), y: (n_classes: 10)
        print('... t_model completed. X (dim: %s), y: (n_classes: %d)' % (str(X.shape), len(np.unique(y)) if y is not None else 1))
    
    m, M = 0.0, 1.0
    sample_ratios = [0.9, 0.1, ] # 90% for training (which is to be subsetted if too big), 10% for model selection
    if doClassify(): 
        # binary classification 
        # t_binary_classify(**kargs)

        # multiclass classification 
        print('test> meta: %s' % tsHandler.meta)
        if tsHandler.is_sparse(): # check tsHandler.d2v
            m, M = t_classify(X=X, y=y, mode='multiclass', n_per_class=nPerClass, tset_dtype='sparse', drop_ctrl=drop_ctrl,
                clf_name='gradientboost') 
            # t_classify(mode='multiclass', n_per_class=nPerClass, tset_dtype='sparse', drop_ctrl=drop_ctrl,
            #     clf_name='gradientboost') 
        else: 
            # load training set from the saved file
            X=y=None; gc.collect()  # get training set from disk
            if tNNet: 
                # import dnn_utils for example networks
                # sample_ratios = [0.1, 0.05, ]  # training/test, validation ... only use a fraction of the total sample size
                epochs_ms, batch_size_ms = 100, 32
                epochs, batch_size = 1000, 16

                m, M = t_deep_classify(mode='multiclass', n_per_class=nPerClass, tset_dtype='dense', drop_ctrl=drop_ctrl, 
                           last_n_visits=n_timesteps, 

                           clf_name='lstm',

                           # this is only used for specifying the fraction of data for model selection
                           ratios=[],  # set to empty list to bypass model selection; # e.g. 10% for training, 5% for validation, the rest ignored
                           ratio = 0.7, # used to specify the ratio of data used for training when N is small (< 10K)
                           
                           model_selection=False, 
                           epochs_ms=epochs_ms, batch_size_ms=batch_size_ms,   # other params: patience_ms,
                           patience_ms=50,

                           n_trials=1, 
                           patience=200, 
                           epochs=epochs, batch_size=batch_size)
            else: 
                m, M = t_classify(mode='multiclass', n_per_class=nPerClass, tset_dtype='dense', drop_ctrl=drop_ctrl, 
                    ratios=sample_ratios,  # ... 10% for training, 5% for validation, the rest ignored
                    clf_name='gradientboost') # don't use 'classifier_name'

    ### re-labeing 
    # t_label(**kargs)

    return (m, M)  # min and max performance scores (e.g. AUC)



# Glcs 

def test(**kargs): 
    def get_global_cohort_dir(name='CKD'): 
        return seqparams.getCohortGlobalDir(name) # sys_config.read('DataExpRoot')/<cohort> 

    import seqConfig as sq

    # params
    # topn
    #     - load documents: 
    #         cohort, seq_ptype, ctype, ifiles, doc_filter_policy, min_ncodes, simplify_code
    #     - load LCS set 
    #         cohort, lcs_type, lcs_policy, ctype, consolidate_lcs, label, meta
    
    userFileID = meta = None  # lcs_policy, (per-class) sample size
    identifier = 'Flcs-fs' # use this for classification results instead of userFileID for now, feature selection
    cohort = kargs.get('cohort', 'CKD')
    # use 'regular' for creating feature set but separate 'diag' and 'med' when doing LCS analysis
    ctype = seqparams.normalize_ctype('regular')  # 'diag', 'med'
    sq.sysConfig(cohort=cohort, seq_ptype=ctype, 
        lcs_type='global', lcs_policy='df', 
        consolidate_lcs=True, 
        slice_policy=kargs.get('slice_policy', 'noop'), 

        simplify_code=kargs.get('simplify_code', False), 
        meta=userFileID)

    tCompleteData = True
    maxNPerClass = None # 10000  # None to include all
    maxNSamplePerClass = None # 3000
    topNFSet = 10000 # 10000

    ### make training set based on path analysis (e.g. LCS labeling)
    # t_model(make_tset=True, make_lcs_feature_tset=False, make_lcs_tset=False)

    ### classification 
    # t_classify(**kargs)

    ### global LCS given a cohort-specific documents 
    # t_analyze_lcs(cohort='CKD', seq_ptype='regular')
    # t_analyze_lcs(cohort='CKD', seq_ptype='diag')
    # t_analyze_lcs(cohort='CKD', seq_ptype='med')

    ### stratified LCS patterns
    inputdir = get_global_cohort_dir(name='CKD')  # large cohort: CKD, dev cohort (n=2833): CKD0
    # t_lcs(inputdir=inputdir)  # derive local LCSs from within each stratum

    ### global -> stratified LCS patterns
    t_lcs2()

    ### create LCS feature set (followed by logistic regression and observe which LCSs in each stratum have relatively higher coefficients)
    # t_model(cohort='CKD', seq_ptype='regular', make_lcs_feature_tset=True, make_lcs_tset=False, make_tset=False)
    # fset = t_classify_lcs(**kargs)

    ### create LCS feature set given LCSs derived from t_lcs2()

    # t_analyze_mcs() # precompute LCSs first i.e. run analyzeMCS()
    # t_lcs_feature_select()  # after running incremental mode of analyzeMCS(), choose from among the LCS candidates
    t_make_lcs_fset() # t_lcs2a()

    tLoadSelectedLCS, tApplyFS = True, True
    if tLoadSelectedLCS: tApplyFS=True
    t_classify(identifier=identifier, 
        clf_name='gbt',
        n_per_class=maxNPerClass,  # classification
        n_sample_per_class=maxNSamplePerClass, # feature selection (used only when apply_fs is True)
        n_features=topNFSet, apply_fs=tApplyFS, load_selected_lcs=tLoadSelectedLCS, 
        drop_ctrl=True)  

    # exploratory cluster analysis 
    # t_cluster()  # feature selection -> frequency analaysis by plotting 2D comparison plots
    
    ### time series of LCSs + sequence colormap
    # t_lcs_timeseries()

    return


# params: cohort, seq_ptype, ifiles, doc_filter_policy
        #         min_ncodes, simplify_code
        # use the derived MCS set (used to generate d2v training data) as the source

else: 
        X, y = kargs.get('X', None), kargs.get('y', None)
        if X is None or y is None: 
            # load, (scale), modify, subsample
            X, y = loadSparseTSet(scale_=kargs.get('scale_', True), 
                                    label_map=seqparams.System.label_map, 
                                    drop_ctrl=tDropControlGroup, 
                                    n_per_class=maxNPerClass)  # subsampling
        else: 
            y = mergeLabels(y, lmap=seqparams.System.label_map) # y: np.ndarray
            # subsampling 
            if maxNPerClass:
                y = np.array(y)
                X, y = subsample(X, y, n=maxNPerClass)

            if kargs.get('scale_', True): 
                # scaler = StandardScaler(with_mean=False)
                scaler = MaxAbsScaler()
                X = scaler.fit_transform(X)

        assert X is not None and y is not None

        ### model selection
        X_val = y_val = X_test = y_test = None  # set aside a small subset of (X, y) for model selection
        if maxSampleRatios: 
            n0 = X.shape[0]
            ridx = sampling.splitDataPerClass(y, ratios=maxSampleRatios)

            # prepare a separate validation set for model selection if possible
            if len(maxSampleRatios) >= 2: 
                X_val, y_val = X[ridx[1]], y[ridx[1]] 

            X, y = X[ridx[0]], y[ridx[0]]   # training; X_test, y_test is included within (X, y)

            ridx = None
            n1 = X.shape[0]
            print('d_classify-sparse> sample size %d (r=%s)=> %d' % (n0, maxSampleRatios, n1))

            if X_val is not None: 
                n2 = X_val.shape[0]
                print('... validation set size %d' % n2)

        scoring_metric = 'neg_log_loss'
        for i, (clf, param_grid) in enumerate(clf_list): 
            if (X_val is not None and y_val is not None) and len(param_grid) > 0:  # if not, just assume clf has been optimally configured
                print('... model selection on classifier: %s | metric: %s' % (clf, ))
                clf = ms.selectModel(X_val, y_val, estimator=clf, param_grid=param_grid, n_folds=5, scoring=scoring_metric)
            clf_list[i] = clf  

        # [test]
        # clf_list = [LogisticRegression(class_weight='balanced', solver='saga', penalty='l1'), ]
        summary(X=X, y=y)
        result_set = []
        for clf in clf_list: 
            res = modelEvaluateBatch(X, y, 
                    classifier=clf, 

                    n_trials=nTrials, 
                    roc_per_class=classesOnROC,
                    # label_map=seqparams.System.label_map, # use sysConfig to specify
                    meta=userFileID, identifier=None)

            # res = multiClassEvaluateSparse(X=X, y=y, 
            #         classifier=clf, 
            #         # classifier_name='l1_logistic',   # if not None, classifier_name takes precedence over classifier
            #         focused_labels=focusedLabels, 
            #         roc_per_class=classesOnROC,
            #         param_grid=param_grid, 
            #         label_map=seqparams.System.label_map, # use sysConfig to specify
            #         meta=userFileID, identifier=None) # use meta or identifier to distinguish different classificaiton tasks; set to None to use default
            result_set.append(res)

# ROC train test split mode 
  roc_train_test_split() … (A)

# subsampling
n0 = X.shape[0]
            ridx = sampling.splitDataPerClass(y, ratios=maxSampleRatios)
            X, y = X[ridx[0]], y[ridx[0]]; ridx = None
            n1 = X.shape[0]
            print('d_classify> sample size %d (r=%s)=> %d' % (n0, maxSampleRatios, n1))



from scipy import interp
    from itertools import cycle
    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    from sklearn import preprocessing

    yl = np.unique(y)  # (unique) class labels
    class_labels = np.unique(y)
    n_classes = len(class_labels)
    n_folds = kargs.get('n_folds', 5)
    random_state = np.random.RandomState(0)

    # transform class labels (y) into numerical values + binarize 
    # [memo] binarized labels don't seem to work with cv.split()
    # y, lookup = transform_label()  # y: binarized label; lookup: numeric -> label
    lb, lookup = encode_labels() 


from sklearn.preprocessing import StandardScaler, MaxAbsScaler # feature scaling for sparse repr. 
from system import utils as sysutils

        if kargs.get('scale_', True): 
            # scaler = StandardScaler(with_mean=False)
            scaler = MaxAbsScaler()
            X = scaler.fit_transform(X)


def remove_classes(X, y, labels=[], other_label='Others'):
        exclude_set = labels if len(labels)>0 else [other_label, ]
        
        N0 = len(y)
        ys = Series(y)
        cond_pos = ~ys.isin(exclude_set)

        idx = ys.loc[cond_pos].index.values
        y = ys.loc[cond_pos].values 
        X = X[idx]  # can sparse matrix be indexed like this?

        print('remove_labels> size(ts): %d -> %d' % (N0, X.shape[0]))        
        return (X, y)

        if maxNPerClass:
            y = np.array(y)
            X, y = subsample(X, y, n=maxNPerClass)

from seqparams import Pathway
        if save_: 
            # this calls saveDataFrame with special keyword padded to 'meta' (Rlcs)
            TSet.saveRankedLCSFSet(df, cohort=cohort, d2v_method=lcsHandler.d2v_method, 
                seq_ptype=lcsHandler.ctype, suffix=lcsHandler.meta) # dir_type/'combined'
       
        # sorted_ = sorted([(lcs, len(dx)) for lcs, dx in lcsToDocIDs], key=lambda x: x[1], reverse=True) # descending order
        return df


    def stratify_docs(inputdir=None, lmap=None): # params: cohort
        ### load + transfomr + (ensure that labeled_seq exists)

        # use the derived MCS set (used to generate d2v training data) as the source
        src_dir = get_global_cohort_dir() if inputdir is None else inputdir # orginal source: get_global_cohort_dir()
        assert os.path.exists(src_dir), "Invaild MCS source dir:\n%s\n" % src_dir
        ifiles = kargs.get('ifiles', ['condition_drug_labeled_seq-CKD.csv', ])
        ctype =  kargs.get('seq_ptype', lcsHandler.ctype)  # kargs.get('seq_ptype', 'regular')
        # cohort = kargs.get('cohort', 'CKD')  # 'PTSD'
        
        minDocLength = kargs.pop('min_ncodes', 10)
        tSimplified = kargs.get('simplify_code', lcsHandler.is_simplified)
        slicePolicy = kargs.get('splice_policy', lcsHandler.slice_policy)
        
        if lmap is None: lmap = policy_relabel() 
        stratified = stratifyDocuments(cohort=cohort, seq_ptype=ctype,
                    predicate=kargs.get('predicate', None), 
                    simplify_code=kargs.get('simplify_code', tSimplified), 

                    # source
                    inputdir=src_dir, 
                    ifiles=kargs.get('ifiles', []), 
                    min_ncodes=minDocLength, 

                    # relabeling operation 
                    label_map=lmap,  # noop for now

                    # slice operations {'noop', 'prior', 'posterior', }
                    slice_policy=slicePolicy, 
                    slice_predicate=kargs.get('slice_predicate', None), 
                    cutpoint=kargs.get('cutpoint', None),
                    inclusive=True)

        ### subsampling
        # maxNDocs = kargs.get('max_n_docs', None)
        # if maxNDocs is not None: 
        #     nD0 = len(D)
        #     D, L, T = sample_docs(D, L, T, n=maxNDocs)
        #     nD = len(D)

        nD = nT = 0
        for label, entry in stratified.items():
            Di = entry['sequence']
            Ti = entry['timestamp']
            # Li = entry['label']
            nD += len(Di)
            nT += len(Ti)

        # Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing
        print('stratify_docs> nD: %d | cohort=%s, ctype=%s, simplified? %s' % (nD, lcsHandler.cohort, lcsHandler.ctype, lcsHandler.is_simplified))
        return stratified


    def load_ranked_fset(topn=None): 
        # load 
        df = TSet.loadRankedLCSFSet(cohort=cohort, d2v_method=lcsHandler.d2v_method, 
                     seq_ptype=lcsHandler.ctype, suffix=lcsHandler.meta)
        if df is not None and not df.empty: 
            if topn is not None: df = df.head(topn)
        return df  # dataframe or None (if not found)

    # policy A: rank feature by specified criteria e.g. term frequency
    df = load_ranked_fset(topn=topNFSet)
    if df is None or df.empty: 
        df = rank_fset(lcsCandidates, analyze_mcs=ret, save_=True, topn=topNFSet) 

    def save_tset(X, y):
        # file ID: cohort, d2v_method, seq_ptype, index, suffix
        # directory: cohort, dir_type/'combined'
        if not isinstance(y, np.ndarray): y = np.array(y)
        TSet.saveSparseLCSFeatureTSet(X, y=y, cohort=cohort, d2v_method=lcsHandler.d2v_method, 
            seq_ptype=lcsHandler.ctype, suffix=lcsHandler.meta)

        # PS1: feature set is defined according to load_rank_fset (precomputed variables) or rank_fset (new variables)

        # PS2: load method
        # X, y = TSet.loadSparse(cohort, **kargs)
        return


    def get_global_cohort_dir(): # source MDS
        return seqparams.getCohortGlobalDir(cohort) # sys_config.read('DataExpRoot')/<cohort>

    def load_ranked_fset(topn=None): 
        # load 
        df = TSet.loadRankedLCSFSet(cohort=cohort, d2v_method=lcsHandler.d2v_method, 
                     seq_ptype=lcsHandler.ctype, suffix=lcsHandler.meta)
        if df is not None and not df.empty: 
            if topn is not None: df = df.head(topn)
        return df  # dataframe or None (if not found)

    def initvarcsv2(name, keys=None, value_type='list', inputdir=None, content_sep=','): 
        # Same as initvar but load lcs map contents from .csv files
        if inputdir is None: inputdir = Pathway.getPath(cohort=kargs.get('cohort', lcsHandler.cohort))
        ctype = seqparams.normalize_ctype(kargs.get('seq_ptype', lcsHandler.ctype))  # 'diag', 'med'

        fname = '%s-%s.csv' % (name, ctype)
        fpath = os.path.join(inputdir, fname)

        newvar = {}
        if keys is not None: 
            if value_type == 'list': 
                newvar = {k: [] for k in keys}
            else: 
                # assuming nested dictionary 
                newvar = {k: {} for k in keys}
        else: 
            if value_type == 'list': 
                newvar = defaultdict(list)
            else: 
                newvar = defaultdict(dict) # nested dictionary

        if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
            df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
            print('initvarcsv2> var: %s > found %d existing entries from %s' % (name, df.shape[0], fname))
            
            # parse 
            if name == 'lcsmap':  # lcs -> docIDs
                header = ['lcs', 'doc_ids']
                idx = []
                for idstr in df['doc_ids'].values: 
                    idx.append(idstr.split(content_sep))
                newvar.update(dict(zip(df['lcs'], idx)))
                return newvar 
            elif name == 'lcsmapInvFreq': 
                header = ['doc_id', 'lcs', 'freq']  # where doc_id can have duplicates 
                # for di in df['doc_id'].unique(): 
                cols = ['lcs', ]
                adict = {}
                for lcs in set(df['lcs'].values): 
                    adict[lcs] = sum(df.loc[df['lcs']==lcs]['freq'].values)
                print('initvarcsv2> Found %d entries ...' % len(adict))     

            else: 
                raise NotImplementedError

        if len(newvar) > 0: 
            print('initvarcsv> example:\n%s\n' % sysutils.sample_dict(newvar, n_sample=1))
        return newvar


    # sample only a subset of the documents [todo]
    tsDType = kargs.get('tset_dtype', 'sparse')
    if tsDType.startswith('d'):  # dense format
        # set a maximum number of tokens to reduce memory requirement
        X = []
        for i, doc in enumerate(docs):
            vec = np.zeros(nF, dtype="float32")
            X.append(to_freq_vector(doc, vec)) # sortedTokens 
            if i % 100 == 0: test_vector(doc, vec)
        return np.array(X)

    # sparse format by default 
    row, col, data = [], [], []
    for i, doc in enumerate(docs):
        idx, cnts = get_counts(doc, sortedTokens)  # column indices, counts
        row.extend([i] * len(idx))  # ith document
        col.extend(idx)  # jth attribute
        data.extend(cnts)  # count 
    Xs = csr_matrix((data, (row, col)), shape=(nD, nF))
    print('getDocVecBow> Found %d coordinates, %d active values, ' % (len(data), Xs.nnz))



    def process_docs(inputdir=None): # params: cohort
        ### load + transfomr + (ensure that labeled_seq exists)

        # use the derived MCS set (used to generate d2v training data) as the source
        src_dir = get_global_cohort_dir() if inputdir is None else inputdir # orginal source: get_global_cohort_dir()
        assert os.path.exists(src_dir), "Invaild MCS source dir:\n%s\n" % src_dir
        ctype =  kargs.get('seq_ptype', 'regular')  # kargs.get('seq_ptype', 'regular')
        # cohort = kargs.get('cohort', 'CKD')  # 'PTSD'

        minDocLength = kargs.pop('min_ncodes', 10)
        tSimplified = kargs.get('simplify_code', False)
        D, L, T = processDocuments(cohort=cohort, seq_ptype=ctype, 
                    inputdir=src_dir, 
                    ifiles=kargs.get('ifiles', []),
                    # meta=kargs.get('meta', None), 
                    
                    # document-wise filtering 
                    policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  
                    min_ncodes=minDocLength,  # retain only documents with at least n codes

                    # content modification
                    # predicate=kargs.get('predicate', None), # reserved for segment_docs()
                    simplify_code=tSimplified, 

                    source_type='default', 
                    create_labeled_docs=False)  # [params] composition

        maxNDocs = kargs.get('max_n_docs', None)
        if maxNDocs is not None: 
            nD0 = len(D)
            D, L, T = sample_docs(D, L, T, n=maxNDocs)
            nD = len(D)

        # Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing
        print('  + nD: %d | cohort=%s, ctype=%s, labeled? %s, simplified? %s' % (len(D), cohort, ctype, is_labeled_data(L), tSimplified))
        return (D, L, T)
    def stratify_docs(inputdir=None, lmap=None): # params: cohort
        ### load + transfomr + (ensure that labeled_seq exists)

        # use the derived MCS set (used to generate d2v training data) as the source
        src_dir = get_global_cohort_dir() if inputdir is None else inputdir # orginal source: get_global_cohort_dir()
        assert os.path.exists(src_dir), "Invaild MCS source dir:\n%s\n" % src_dir
        ifiles = kargs.get('ifiles', ['condition_drug_labeled_seq-CKD.csv', ])
        ctype =  kargs.get('seq_ptype', 'regular')  # kargs.get('seq_ptype', 'regular')
        # cohort = kargs.get('cohort', 'CKD')  # 'PTSD'
        
        minDocLength = kargs.pop('min_ncodes', 10)
        tSimplified = kargs.get('simplify_code', False)
        
        if lmap is None: lmap = policy_relabel() 
        stratified = stratifyDocuments(cohort=lcsHandler.cohort, seq_ptype=lcsHandler.ctype,
                    predicate=kargs.get('predicate', None), 
                    simplify_code=kargs.get('simplify_code', lcsHandler.is_simplified), 

                    # source
                    inputdir=src_dir, 
                    ifiles=kargs.get('ifiles', []), 
                    min_ncodes=minDocLength, 

                    # relabeling operation 
                    label_map=lmap, 


                    # slice operations {'noop', 'prior', 'posterior', }
                    slice_policy=lcsHandler.slice_policy, 
                    slice_predicate=kargs.get('slice_predicate', None), 
                    cutpoint=kargs.get('cutpoint', None),
                    inclusive=True)

        ### subsampling
        # maxNDocs = kargs.get('max_n_docs', None)
        # if maxNDocs is not None: 
        #     nD0 = len(D)
        #     D, L, T = sample_docs(D, L, T, n=maxNDocs)
        #     nD = len(D)

        nD = nT = 0
        for label, entry in stratified.items():
            Di = entry['sequence']
            Ti = entry['timestamp']
            # Li = entry['label']
            nD += len(Di)
            nT += len(Ti)

        # Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing
        print('stratify_docs> nD: %d | cohort=%s, ctype=%s, simplified? %s' % (nD, lcsHandler.cohort, lcsHandler.ctype, lcsHandler.is_simplified))
        return stratified
    def get_tset_dir(): # derived MDS 
        # params: cohort, dir_type
        tsetPath = TSet.getPath(cohort=kargs.get('cohort', lcsHandler.cohort), dir_type=kargs.get('dir_type', 'combined'))  # ./data/<cohort>/
        print('t_lcs2> training set dir: %s' % tsetPath)
        return tsetPath 
    def get_global_cohort_dir(): # source MDS
        return seqparams.getCohortGlobalDir(kargs.get('cohort', lcsHandler.cohort)) # sys_config.read('DataExpRoot')/<cohort>



# pathAnalyzer.
def markTSetByLCS(cohort, **kargs):

    for j, lcs in enumerate(df['lcs'].values): # Pathway.header_global_lcs = ['lcs', 'length', 'count', 'n_uniq']
        # linear search (expansive): how many patients contain this LCS? 
        if not lcsmap.has_key(lcs): lcsmap[lcs] = []
        lcs_seq = lcs.split(lcs_sep)  # Pathway.strToList(lcs)  

        # [test]
        # if j == 0: print('  + tokenized LCS: %s' % lcs_seq)  # ok. 

        n_subs = 0
        for i, doc in enumerate(D):  # condition: each 'seq' is a list of strings/tokens
            if len(lcs_seq) > len(doc): continue 
            if seqAlgo.isSubsequence(lcs_seq, doc): # if LCS was derived from patient doc, then at least one match must exist
                
                # find corresponding timestamps 
                # lcs_tseq = lcs_time_series(lcs_seq, i, D, T) # [output] [(<code>, <time>), ...]
                lcsmap[lcs].append(i)  # add the person index
                lcsmapInv[i].append(lcs)
                matched_docIDs.add(i)  # global 
                n_subs += 1 
        n_personx.append(n_subs)  # number of persons sharing the same LCS

        # subsetting document IDs
        if maxNIDs is not None and len(lcsmap[lcs]) > maxNIDs: 
            lcsmap[lcs] = random.sample(lcsmap[lcs], min(maxNIDs, len(lcsmap[lcs])))

    return

### old labeled training set (diabetes)
def make_tset_labeled(...)
    # d2v_method = 'tfidfavg'
    seq_ptype_eff = 'regular'
    identifier = '%s-%s-%s' % (seq_ptype_eff, w2v_method, d2v_method)

    if cohort_name is None or cohort_name.startswith('dia'):  # [old]

        # cohort: diabetes, n_classes 2 or 3
        # labeling convention: type 1: 0, type 2: 1, type 3 (gestational): 2
        t1idx, t2idx, t3idx = res['type_1'], res['type_2'], res['gestational']  # map: classes -> indices
        # t1idx = res.get(0, 'type_1')
        # t2idx = res.get(1, 'type_2')

        # [params] file identifier 
        fpath = os.path.join(basedir, 'tset_nC%s_ID%s-G%s.csv' % (n_classes, identifier, cohort_name))
        ts = load_fset(fpath, fsep=fsep)  # [I/O]

        # [generic] vecto.getFeatureVecs(docs, model, n_features, **kargs)
        if ts is None:  
        
            # X = vector.getTfidfAvgFeatureVecs(sequences, model, n_features) # optional: min_df, max_df, max_features
            X = vector.getFeatureVectors(docs, model, n_features, d2v_method=d2v_method)
            assert X.shape[0] == n_doc0

            X_t1, y_t1 = X[t1idx], np.repeat(0, len(t1idx))  # type 1 is labeled as 0

            X_t2, y_t2 = X[t2idx], np.repeat(1, len(t2idx))  # type 2 is labeled as 1  
            Xb = np.vstack([X_t1, X_t2])  # b: binary classification
            yb = np.hstack([y_t1, y_t2])
            idxb = np.hstack([t1idx, t2idx])
            assert Xb.shape[0] == len(yb)
            n_docb = Xb.shape[0]

            print('output> preparing ts for binary classification > method: %s, n_doc: %d' % (doc2vec_method, n_docb))
            header = ['%s%s' % (f_prefix, i) for i in range(Xb.shape[1])]
            ts = DataFrame(Xb, columns=header)
            ts[TSet.target_field] = yb
            ts[TSet.index_field] = idxb
            ts = ts.reindex(np.random.permutation(ts.index)) 

            # fpath = os.path.join(basedir, 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc0, seq_ptype)) 
            print('output> saving (binary classification, d2v method=%s) training data to %s' % (doc2vec_method, fpath))
            ts.to_csv(fpath, sep=',', index=False, header=True) # cmp: previously, we use ts.index to keep labels => index=True 
    else: # [new] cohorts other than the default diabetes
        fpath = os.path.join(basedir, 'tset_nC%s_ID%s-G%s.csv' % (n_classes, identifier, cohort_name))
        ts = load_fset(fpath, fsep=fsep)  # [I/O]
        
        if ts is None: 
            X = vector.getFeatureVectors(docs, model, n_features, d2v_method=d2v_method)
            assert X.shape[0] == n_doc0

        n_classes_prime = len(res)
        print('verify> requested n_classes: %d, given: %d' % (n_classes, n_classes_prime))
        tss = []
        # for nc in range(n_classes_prime): 
        for label, idx in res.items(): 
            # idx = res.get(nc, res.get('class_%s' % nc, []))  # res: class label -> index set

            if len(idx): 
                Xs, ys = X[idx], np.repeat(label, len(idx)) 
                header = ['%s%s' % (f_prefix, i) for i in range(Xs.shape[1])]
                tu = DataFrame(Xs, columns=header)
                tu[TSet.target_field] = ys
                tu[TSet.index_field] = idx  # keep track of indices into X
                tu = tu.reindex(np.random.permutation(tu.index))
                tss.append(tu)

        if len(tss) > 0: 
            ts = pd.concat(tss, ignore_index=True)
            print('output> saving (%d-class classification, d2v method=%s) training data to %s' % (n_classes, doc2vec_method, fpath))
            ts.to_csv(fpath, sep=',', index=False, header=True) # cmp: previously, we use ts.index to keep labels => index=True 
        else: 
            print('output> No data found!')

    return ts


    def get_sources():
        docfiles = seqparams.TDoc.getPaths(cohort=cohort_name, basedir=basedir, doctype='timed', 
            ifiles=kargs.get('ifiles', []), verfiy_=True)  # if ifiles is given, then cohort is ignored
        return docfiles   

    def get_sources():
        fp0 = kargs.get('ifile', None)
        if fp0 is not None: 
            assert isinstance(fp0, str)
            # does it contain a root dir? 
            basedir0, fname0 = os.path.dirname(fp0), os.path.basename(fp0)
            if not basedir0: basedir0 = basedir 
            fp0 = os.path.join(basedir0, fname0)

            docfiles0 = [fp0, ]
        else: 
            fp0 = os.path.join(basedir, seqparams.TDoc.getName(cohort=cohort_name, doctype='timed'))
            docfiles0 = [fp0, ]

        docfiles = kargs.get('ifiles', docfiles0)  # if given, must be full paths
       
        # normalize 
        for i, f in enumerate(docfiles): 
            # contain rootdir? 
            rootdir, fname0 = os.path.dirname(f), os.path.basename(f) 
            if not rootdir: rootdir = basedir
            fp = os.path.join(rootdir, fname0)
            assert os.path.exists(fp), "Invalid input source file: %s" % fp
            ifiles[i] = fp

        print('read> input files:\n%s\n' % docfiles) 

        # verfiy the source(s)
        for fp in docfiles: 
            assert fp.find('timed_seq') > 0
            assert os.path.exists(fp), "Invalid coding sequence source path: %s" % fp    

        return docfiles   

def data_matrix(**kargs):  # [precond] t_preclassify() or make_tset() has been called
    """
    Read training data from files where each file contains 
    (doc) feature vectors and (surrogate) labels derived from pre_classify with surrogate labeling and pre-computed d2v 

    A wrapper of load_tset()

    """
    from seqparams import TSet
    # import seqAnalyzer as sa

    # [params] read the document first followed by specifying training set based on seq_ptype and d2v method
    cohort_name = kargs['cohort'] # kargs.get('cohort', 'diabetes')

    # outputdir = os.path.join(os.getcwd(), 'data')  # global data dir: sys_config.read('DataExpRoot')
    basedir = outputdir = seqparams.get_basedir(cohort=cohort_name) 

    read_mode = kargs.get('read_mode', 'doc')  # doc, random, etc. 
    seq_ptype = seqparams.normalize_ptype(**kargs)

    w2v_method = word2vec_method = kargs.get('w2v_method', 'sg')
    d2v_method = doc2vec_method = kargs.get('d2v_method',  'tfidfavg') # PVDM
    n_classes = seqparams.arg(['n_classes', 'n_labels'], None, **kargs)

    tset_version = 'new'  # or 'old'
    tset_type = kargs.get('tset_type', 'binary') 
    if n_classes is not None: seqparams.normalize_ttype(n_classes)

    subset_idx = seqparams.arg(['idx', 'subset_idx'], None, **kargs) # training set subsetting
    fsep = ','
    tset_stem = 'tset'
    load_seq, save_seq = False, True
    
    ### read coding sequences
    print('io> reading coding sequences from source ...')
    sequences = sa.read(simplify_code=False, mode=read_mode, verify_=False, seq_ptype=seq_ptype, cohort=cohort_name, 
                           load_=load_seq, save_=save_seq)
    n_doc = len(sequences)
    # but this loads all sequences!

    ts = None
    if tset_version == 'old':  # [obsolete][hardcode]
        if cohort_name is None or cohort_name.startswith('diab'): 
            fpath_default = os.path.join(basedir, 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc, seq_ptype)) 
            if tset_type.startswith('bin'): 
                print('io> loading binary class training data ... ')
                fpath = os.path.join(basedir, 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc, seq_ptype)) 
        
            elif tset_type.startswith(('multi', 't')):
                print('io> loading 3-label multiclass training data ... ')
                fpath = os.path.join(basedir, 'tset_%s_%s_t1t2t3-P%s.csv' % (doc2vec_method, n_doc, seq_ptype)) 
            else: 
                print('io> unknown tset_type: %s > use %s by default!' % (tset_type, os.path.basename(fpath_default)))
                fpath = fpath_default
        else: 
            raise NotImplementedError
            
        if os.path.exists(fpath): 
            ts = pd.read_csv(fpath, sep=fsep, header=0, index_col=False, error_bad_lines=True)
            print('output> loaded training data | d2v_method: %s > tset dim: %s' % (doc2vec_method, str(ts.shape)))

    else: # new version
        ts = load_tset(w2v_method=w2v_method, d2v_method=d2v_method, 
                        seq_ptype=seq_ptype, read_mode=read_mode, 
                            cohort=cohort_name, idx=subset_idx)  # subset_idx: subsetting training data by indices
        if ts is None: 
            print('status> No training set found! Making one ...')

            # [note] test_model is for sa.loadModel, which enables the utility of learned w2v repr (similarities etc)
            ts = make_tset(w2v_method=w2v_method, d2v_method=d2v_method, 
                                seq_ptype=seq_ptype, read_mode=read_mode, test_model=False, 
                                cohort=cohort_name, idx=subset_idx) # [note] this shouldn't revert back and call load_tset() again
    assert ts is not None and not ts.empty, 'data_matrix> Warning: No training set found!'

    if ts.shape[0] < n_doc: 
        idx = ts[TSet.index_field].values # ~ positions of sequences
        D = sa.select(docs=sequences, idx=idx)
        print('stats> subset original docs from size of %d to %d' % (n_doc, len(D)))
    elif ts.shape[0] == n_doc: 
        D = sequences
    else: 
        raise ValueError, "Number of documents: %d < training set size: %d ??" % (n_doc, ts.shape[0])

    return (D, ts)



    if kargs.get('load_model', True) and os.path.exists(fpath):
        model = Word2Vec.load(fpath)  # can continue training with the loaded model!

        # [note] can do incremental training
        # model.train(more_sentences)

    else: 
        # [todo] write a separate module for different modalities
        # 1. document embedding 2. seq-To-seq LSTM, etc. 

        labeled_docs = [] # used only for D2V

        # [note] default = 1 worker = no parallelization; for workers to work, need cython installed
        print('vector> computing w2v model (w2v (sg=%d): %s, n_features: %d, window: %d, ctype: %s)' % \
            (sg, w2v_method, n_features, window, seq_ptype))

        # [params] gensim
        # 1. sg defines the training algorithm. By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.
        # 2. negative = if > 0, negative sampling will be used, the int for negative specifies how many “noise words” 
        #    should be drawn (usually between 5-20). Default is 5. If set to 0, no negative samping is used.
        if sg == 1 or w2v_method == 'sg': 
            # [todo] choose which negative sampls to use
            model = Word2Vec(sequences, sg=1, size=n_features, negative=5, window=window, min_count=min_count, workers=n_workers)  # [1]
        elif sg == 0 or w2v_method == 'cbow': 
            model = Word2Vec(sequences, sg=0, size=n_features, negative=5, window=window, min_count=min_count, workers=n_workers) 
        elif w2v_method == 'pv-dm': # really is d2v_method  

            tUseD2V = True 
            labeled_docs = makeD2VLabels(sequences=sequences) # labels: use labelDocByFreqDiag() to generate by default

            # PV-DM w/average
            model = Doc2Vec(dm=1, dm_mean=1, dm_concat=0, size=n_features, window=window, negative=5, hs=0, min_count=min_count, workers=n_workers)
            # but each sequence needs to have a label 
            model.build_vocab(labeled_docs) # [error] must sort before initializing vectors/weights

            for epoch in range(10):
                model.train(labeled_docs)
                model.alpha -= 0.002  # decrease the learning rate
                model.min_alpha = model.alpha  # fix the learning rate, no decay    
            
        elif w2v_method == 'pv-dbow': 
            tUseD2V = True
            labeled_docs = makeD2VLabels(sequences=sequences) # labels: use labelDocByFreqDiag() to generate by default

            # PV-DBOW 
            model = Doc2Vec(dm=0, dm_concat=0, size=100, negative=5, hs=0, min_count=min_count, workers=n_workers)
            model.build_vocab(labeled_docs) # [error] must sort before initializing vectors/weights
            for epoch in range(10):
                model.train(labeled_docs)
                model.alpha -= 0.002  # decrease the learning rate
                model.min_alpha = model.alpha  # fix the learning rate, no decay    

        else: 
            raise NotImplementedError, "Unknown w2v method: %s" % w2v_method





trace1 = go.Bar(
    y=['giraffes', 'orangutans', 'monkeys'],
    x=[20, 14, 23],
    name='SF Zoo',
    orientation = 'h',
    marker = dict(
        color = 'rgba(246, 78, 139, 0.6)',
        line = dict(
            color = 'rgba(246, 78, 139, 1.0)',
            width = 3)
    )
)
trace2 = go.Bar(
    y=['giraffes', 'orangutans', 'monkeys'],
    x=[12, 18, 29],
    name='LA Zoo',
    orientation = 'h',
    marker = dict(
        color = 'rgba(58, 71, 80, 0.6)',
        line = dict(
            color = 'rgba(58, 71, 80, 1.0)',
            width = 3)
    )
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='marker-h-bar')



                        gfreq = global_motifs[n][ngr] 
                assert gcount >= count, "n-gram %s: local/cluster count: %d > global count: %d ???" % (str(ngr), count, gfreq)  
                tfreq, dfreq = count, len(idf_stats[ngr])  # term frequency, document/cluster frequency



    import icd9utils

    alist = ['296.30', 'F33.9', '311', '296.20', ]

    header = ['code', 'med', 'description']
    adict = {h: [] for h in header}
    for code in alist: 
        medcode = ICD9ToMed(code)
        print('  + %s -> med: %s' % (code, medcode))

        if medcode is None: 
            name = icd9utils.getName(code)
        else: 
            name = getName2(medcode)
        print('  + %s -> desc: %s' % ('?' if medcode in (None, '') else medcode, name))

        adict['code'].append(code)
        adict['med'].append(medcode)
        adict['description'].append(name)

eval_motif
  algorithms.count_ngrams2(seqx, min_length=1, max_length=ng_max, partial_order=partial_order) 

ts = make_tset(w2v_method=w2v_method, d2v_method=d2v_method, 
                        seq_ptype=seq_ptype, read_mode=read_mode, 
                            cohort=cohort_name, idx=subset_idx)

make_tset_unlabeled(w2v_method=w2v_method, d2v_method=d2v_method, 
                            seq_ptype=seq_ptype, read_mode=read_mode, test_model=test_model,
                                cohort=cohort_name, bypass_lookup=bypass_lookup) 

        for cid, ng_cnts in tfidf_stats.items(): 

            n_ng_cnts = {}  # size to ngram to count
            lengths = []
            for ngr, cnt in ng_cnts: 
                n = len(ngr); lengths.append(n)
                if not n_ng_cnts.has_key(n): n_ng_cnts[n] = {}
                n_ng_cnts[n][ngr] = cnt
            
            for n in lengths: # topn for each length
                counter = collections.Counter(n_ng_cnts[n]) 
                req_items = counter.most_common(topn)  # topn are mostly unigrams


            ngr_count, ngr_cids = entries['global'], entries[]
            ngstr = wsep.join(str(e) for e in ngr)  # k: tupled n-gram

            adict['ngram'].append(ngstr)
            adict['cluster_freq'].append(ngr_count)

            labels = []
            for cid in ngr_cids: 
                labels.append(clabels[cid])
                
            cids_str = lsep.join(str(c) for c in ngr_cids)
            labels_str = lsep.join(str(l) for l in labels)
            adict['cids'].append(cids_str)
            adict['labels'].append(labels_str)

        df = DataFrame(adict, columns=header)
        print('io> saving tfidf stats (dim: %s) to %s' % (str(df.shape), fpath))
        df.to_csv(fpath, sep='|', index=False, header=True)


        fname_tfidf = 'tfidf-%s.csv' % identifier # identifier: (tset_type, cluster_method, seq_ptype, d2v_method)
        fpath = os.path.join(outputdir, fname_tfidf) # fname_tdidf(ctype, )
        header = ['ngram', 'cluster_freq', 'global_freq', 'cids', 'labels']
                
        wsep = ' '
        lsep = ','  # separator for list objects 
        adict = {h: [] for h in header}
                
        # cidx, labelx = [], []
        for ngr, entries in tfidf_stats.items(): # ngr: n-gram in tuple repr
            ngr_count, ngr_cids = entries['count'], entries['cids']
            ngstr = wsep.join(str(e) for e in ngr)  # k: tupled n-gram

            adict['ngram'].append(ngstr)
            adict['cluster_freq'].append(ngr_count)

            labels = []
            for cid in ngr_cids: 
                labels.append(clabels[cid])
                
            cids_str = lsep.join(str(c) for c in ngr_cids)
            labels_str = lsep.join(str(l) for l in labels)
            adict['cids'].append(cids_str)
            adict['labels'].append(labels_str)

        df = DataFrame(adict, columns=header)
        print('io> saving tfidf stats (dim: %s) to %s' % (str(df.shape), fpath))
        df.to_csv(fpath, sep='|', index=False, header=True)



    basedir = '/phi/proj/poc7002/tpheno/PTSD'


    # [note] it's possible that members never get to be leaders but we want each member to be its own group
    ugroups = set(conceptToLabs.keys())
    ungrouped = ulabs-ugroups
    for c in ungrouped: 
        conceptToLabs[c] = set([c])
    print('info> added %d ungrouped members (each member is its own group)' % len(ungrouped)) # [log] 1222

    ### step 0: read lab concept grouping file obtained from t_map_lab()
    # [input] e.g. tpheno/data-exp/grouped_labs-Pge3-Tbmeasurement-PTSD.csv
    div(message='Stage 0: Load lab concept mapping file ...')
    fname = 'grouped_labs-PTSD.csv'  # explicit: grouped_labs-Pge3-Tbmeasurement-PTSD.csv 
    header = ['group', 'individual_lab', 'group_description', 'individual_description']
    fpath = os.path.join(basedir, fname)
    
    df_map = pd.read_csv(fpath, sep=',', header=0, index_col=False, error_bad_lines=True)
    print('io> loaded concept grouping (dim: %s): %s' % (str(df.shape), fpath))    


# graph traversal
    heads = set(conceptToLabs) # set of keys (group MED codes)
    conceptToLabsPrime = {} # {cH: set() for cH in conceptToLabs.keys()}
    for cH, codes in conceptToLabs.items(): 
        # conceptToLabsPrime[cH].update(codes)

        others = set(heads)-set([cH])
        is_nested = False

        for cH2 in others: 
            if cH in conceptToLabs[cH2]: # if cH is a member of another leader cH2, then cH < cH2, cH2 should inherit all cH members
                is_nested = True

                # merge: cH2 inherit both the original cH2 members AND also cH's members
                if not conceptToLabsPrime.has_key(cH2): 
                    conceptToLabsPrime[cH2] = set()
                    conceptToLabsPrime[cH2].update(conceptToLabs[cH2])
                conceptToLabsPrime[cH2].update(conceptToLabs[cH])

    heads = set(conceptToLabs) # set of keys (group MED codes)
    conceptToLabsPrime = {} # {cH: set() for cH in conceptToLabs.keys()}
    leaders, members = [], []
    for cH, codes in conceptToLabs.items(): 
        for c in codes:  
            leaders.append(cH) # group representative code
            members.append(c) # group members
    links = {}
    for i, (l, m) in enumerate(zip(leaders, members)): 
        for j, m2 in enumerate(members): 
            if i == j: continue # foreach i <> j
            if l == m2:  
                 links[l] 


    for i, (c1, vals1) in enumerate(ordered_edict):  
        # if c1 in grouped: continue
        
        glabs2[c1] = []  # c1 to its potential class (default is itself)
        # grouped.add(c1)

        for j, (c2, vals2) in enumerate(ordered_edict):  # need full search because order is important (c1, c2) <> (c2, c1) 
            # if c2 in grouped: continue 
            if c2 == c1: continue
                    
            et1, et2 = edict[c1], edict[c2]
            assert len(et1) > 0 and len(et2) > 0

            if et1 == et2:  # or qrymed2.isA()
                s1, s2 = sdict.get(c1, []), sdict.get(c2, [])
                if not s1 and not s2: 
                    print(' + case 1> lab (%s, %s) has same M.E. (%s) but neither has assessed specimens' % (c1, c2, str(et1)))
                    # print('  + linking lab %d to lab %s | %s == %s' % (c1, glabs[c1], str(et1), str(et2)))
                    n_case1 += 1
                elif not s1 or not s2: # one is None but not both
                    print(' + case 2> lab (%s, %s) has same M.E. (%s) but one of them does not have specimens' % (c1, c2, str(et1)))
                    if not s1: 
                        print("      ++ lab %s has no sample vs lab %s has %s" % (c1, c2, str(s2)))
                    else: 
                        print("      ++ lab %s has no sample vs lab %s has %s" % (c2, c1, str(s1)))
                    n_case2 += 1
                else:  # both have specimens 

                    n_case3 += 1 
                    if s1 == s2: 
                        glabs2[c1].append(c2)  # use the first code c1 as the group leader, c1 <= c2
                        print('  + linking lab %d to lab %s | ME: %s == %s, S: %s == %s' % (c1, c2, str(et1), str(et2), str(s1), str(s2) ))
                        n_case3a += 1
                    elif is_subset(s1, s2, bidirection=False): 
                        glabs2[c1].append(c2)
                        print('  + linking lab %d to lab %s | ME: %s == %s, S: %s == %s' % (c1, c2, str(et1), str(et2), str(s1), str(s2) ))
                        n_case3b += 1
                    else: 
                        # [test]
                        if len(s1) > 1 or len(s2) > 1: # tricky case: multiple samples yet not identical
                            div(message='   ++ both lab have multiple samples: (%s: %s VS %s: %s)' % (c1, str(s1), c2, str(s2)))
                            n_case3ax += 1  
                            if len(s1) != len(s2): 
                                div(message='      +++ multiple samples yet not identical! (%s: %s VS %s: %s)' % (c1, str(s1), c2, str(s2)))
                                n_case3axx += 1 

                        # policy #2a
                        # found_isA = False  # loose if found ANY isA
                        # for s1i in s1:
                        #     for s2j in s2: 
                        #         if s1i in sdict_desc[s2j]: # if exists any pair s.t. s1i < s2j (or s1 is-a s2)     
                        #             glabs2[c1].append(c2)
                        #             print('  + linking lab %d to lab %s | ME: %s == %s, S: %s < %s' % (c1, c2, str(et1), str(et2), str(s1), str(s2) )) 
                        #             found_isA = True 
                                
                        #         if found_isA: break
                        #     if found_isA: break

                        # policy #2b 
                        # foreach s1i, can at least find one isA in s2? 
                        n_found_isA = 0
                        for s1i in s1:
                            found_isA_in_s2 = False 
                            for s2j in s2:
                                if s1i in sdict_desc[s2j]: # if exists any pair s.t. s1i < s2j (or s1 is-a s2) 
                                    found_isA_in_s2 = True
                                    break 
                            if found_isA_in_s2: 
                                n_found_isA += 1

                        # [log] (cond=multiple samples) linking lab 32732 to lab 89790 | ME: [31955] == [31955], S: [149] < [2393, 35641] 
                        #       149 isA 2393
                        if n_found_isA == len(s1): # each s1i should at least have a correspdoning s2j, where s1i < s2j
                            glabs2[c1].append(c2)  # c1 < c2
                            print('  + (cond=multiple samples) linking lab %d to lab %s | ME: %s == %s, S: %s < %s' % (c1, c2, str(et1), str(et2), str(s1), str(s2) )) 
                            n_case3c += 1 

            # [todo] it's possible that et1 != et2 and s1 != s2
            # elif is_subset(et1, et2, bidirection=True): 
            #     div(message='  + MEntity(%s) ~ MEntity(%s): %s ~ %s' % (c1, c2, str(et1), str(et2)), symbol='%')
            #     n_me_subset += 1

        ### end foreach lab code c2 
    ### end foreach lab code c1

    # [log]
    # info> n_case1 (no specimens): 82, n_case2 (only one has specimen): 1906, n_case3 (both have): 28112
    # info> n_case3: 28112 | n_case3a (s1=s2): 7490, n_case3b (s1-:s2): 60, n_case3c (s1<s2): 60
    # info> n_case3: 28112 | n_case3ax: 482, n_case3axx: 474
    print('info> n_case1: %d, n_case2: %d, n_case3: %d' % (n_case1, n_case2, n_case3))
    print('info> n_case3: %d | n_case3a (s1=s2): %d, n_case3b (s1-:s2): %d, n_case3c (s1<s2): %d' % (n_case3, n_case3a, n_case3b, n_case3b))
    print('info> n_case3: %d | n_case3ax: %d, n_case3axx: %d' % (n_case3, n_case3ax, n_case3axx))

    # [poll]
    desc_map = {}
    # for _, ancestors in glabls2.items(): 
        
    #     candicate_class = set(ancestors)
    #     for ca in ancestors: 
    #         descx = qrymed2.getDescendent(ca, self_=False, to_int=True)
    n_linked = n_linked_multiclass = 0
    conceptToLabs = {}

    # each code 'c' has potential classes (more general code) cx
    for c, cx in glabs2.items(): # glabs2: code => candidate classes
        if not conceptToLabs.has_key(c): conceptToLabs[c] = set([c])  # every code is at least a concept of itself

        if cx: 
            print('glabs2> code %s was linked to %s' % (c, str(cx)))
            n_linked += 1 

            # unroll
            for cH in cx:  # c < cH foreach cH
                if not conceptToLabs.has_key(cH): conceptToLabs[cH] = set([cH])
                conceptToLabs[cH].add(c)  # c is a cH; don't repeat

            if len(cx) > 1: 
                print('verify> lab %s was linked to multiple classes: %s' % (c, cx))
                n_linked_multiclass += 1

    # cH may be a member of some other groups => need to find the ultimate group leader
    if concept_grouping.startswith('d'):  
        div(message='Consolidating all group leaders that are themselves members of other group leaders ...')

        n_nested = 0
        header = ['group', 'member']
        adict = {h:[] for h in header}

        # build a trace chain first
        heads = set(conceptToLabs) # set of keys (group MED codes)
        conceptToLabsPrime = {} # {cH: set() for cH in conceptToLabs.keys()}
        chains = {}
        for cH, codes in conceptToLabs.items(): 
            # conceptToLabsPrime[cH].update(codes)
            others = heads-set([cH])
            is_nested = False
            for cH2 in others: 
                if cH in conceptToLabs[cH2]: # if cH is a member of another leader cH2, then cH < cH2, cH2 should inherit all cH members
                    is_nested = True
                    if not chains.has_key(cH): chains[cH] = set()
                    chains[cH].add(cH2)  # link cH to cH2 (cH < cH2)
            if is_nested: n_nested += 1

        print('info> Found %d (=?=%d) group leaders nested underneath someone else' % (n_nested, len(chains)))

    # orderedLabs = []
    for cH in conceptToLabs.keys(): 
        n = len(conceptToLabs[cH])
        assert n >= 1 
        conceptToLabs[cH] = sorted(list(conceptToLabs[cH]))  # order
        # orderedLabs.append((cH, sorted(list(conceptToLabs[cH]))))

    # [log] n_linked (codes with assigned classes in glabs2): 1490 | n_linked::multiclass: 1073
    print('policy #2> n_linked (codes with assigned classes in glabs2): %d | n_linked::multiclass: %d' % (n_linked, n_linked_multiclass))

    # save the mapping assuming that dflab_map is avail 
    # [output] io> saving df[dim: (12489, 4) (n_codes: 2089)] of grouped labs to tpheno/data-exp/grouped_labs-Pconcept_hierarchy-Tbmeasurement-PTSD.csv
    overwrite = True
    stem, identify = ('grouped_labs', 'Pconcept_hierarchy-Tb%s-%s' % (table_name, cohort_name))
    fname = 'grouped_labs-P%s-Tb%s-%s.csv' % (concept_grouping, table_name, cohort_name)
    fpath = os.path.join(outputdir, fname)
    header = ['group', 'individual_lab', 'group_description', 'individual_description']
    sortbyattr = ['group', 'individual_lab']
    if overwrite or not os.path.exists(fpath): 
        adict = {h:[] for h in header}
        for cH, codes in conceptToLabs.items(): 
            for c in codes:  
                adict['group'].append(cH) # group representative code
                adict['individual_lab'].append(c) # group members
                adict['group_description'].append(dflab_map[cH])
                adict['individual_description'].append(dflab_map[c])
        df = DataFrame(adict, columns=header)
        # df = df.sort('group', ascending=True)
        df.sort_values(sortbyattr, ascending=True, inplace=True)
        print('io> saving df[dim: %s (n_codes: %d)] of grouped labs to %s' % (str(df.shape), len(conceptToLabs), fpath))
        df.to_csv(fpath, sep='|', index=False, header=True)    
            
    return


    # Policy #2: group labs accordig to Jim's suggested rule 
    div(message='Grouping lab codes according to concept hierarchy of ME and assessed specimens ...')
    ordered_edict = zip(edict.keys(), edict.values())
    grouped = set(); glabs2 = {}
    for i, (c1, meX1) in enumerate(edict.items()):  # for each measured entity (m.e.)
        # if c1 in grouped: continue
        glabs2[c1] = []  # map code to its potential concepts
        ancestors = ancmap[c1]
        sX1 = sdict.get(c1, [])
        
        for j, c1a in enumerate(ancestors): # foreach ancestor of 'c1' (a lab code with m.e.)
            me_matched = s_matched = False

            assert isinstance(c1a, int)
            meX0 = edict0.get(c1a, [])  # measure entities (no need to use superset edict0, because we need '==' to hold)
            if meX0: 
                # meX0 = normalize(meX0)
                if meX0 == meX1: 
                    me_matched = True
                    # glabs2[c1].append(c1a) # c1's ancestor c1a is a potential concept class
            
            # sX0 = sdict.get(c1a, [])  # ancestor specimens
            # if sX1 and sX0: 
            #     # if multiple specimens, then require equality (o.w. too complicated)
            #     if sX1 == sX0: 
            #         s_matched = True 
            #     else: 
            #         if len(sX1) == len(sX0): 
            #             for j, s1 in enumerate(sX1):
            #                 sdict_desc[s1] 
            if me_matched: # and s_matched 
                glabs2[c1].append(c1a) # c1's ancestor c1a is a potential concept class
    
    # [poll]
    desc_map = {}
    # for _, ancestors in glabls2.items(): 
        
    #     candicate_class = set(ancestors)
    #     for ca in ancestors: 
    #         descx = qrymed2.getDescendent(ca, self_=False, to_int=True)
    n_linked = 0
    for c, cx in glabs2.items(): 
        if cx: 
            print('debug> code %s was linked to %s' % (c, str(cx)))
            n_linked += 1 
    print('info> n_linked (codes with assigned classes in glabs2): %d' % n_linked)
    
    # preprocess by absorbing descendents 
    # for c1, ancestors in glabs2.items(): 
    #     pass  

    # # foreach code that has measurement(s), find its ancetors
    # temp_path = os.path.join(basedir, 'ancestor_map.pkl')
    # ansmap = {}  # ancestor map: code -> ancestors 
    # if load_data and (os.path.exists(temp_path) and os.path.getsize(temp_path) > 0): 
    #     mdict = pickle.load(open(temp_path, 'rb'))
    # else: 
    #     for code, entities in edict.items():  # for each code with 'entity measured' 
    #         ancestors = qrymed2.getAncestor(code) # default: to_int=True
    #         if ancestors is not None: 
    #             ansmap[code] = ancestors