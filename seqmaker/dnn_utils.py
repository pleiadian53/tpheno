# encoding: utf-8

# non-interactive mode
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt
plt.style.use('ggplot')

# import numpy
import numpy as np
import os, sys, collections, random, gc, time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# word embedding layer
from keras.layers import Embedding

# MCS modules 
from seqConfig import tsHandler
from tset import TSet
import seqparams

# fix random seed for reproducibility
# seed = 53
# np.random.seed(seed)

# support roc_auc in model evaluation
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping

# model selection 
from sklearn.model_selection import train_test_split


class NNS(object):
    epochs = 200
    batch_size = 16 

    identifier = None

    # [params] model complexity
    # A. early stop
    patience_ms = patience = 20
    epochs_ms = 100
    batch_size_ms = 32   # ms: model selection

    model_name = 'nns'  # base name of the model (to which an idenifier is to be attached)

    # weight initialization 
    init_weights = 'init_weights.h5'

    # score indices 
    score_map = {0:'loss', 1:'acc', 2:'auc_roc'} 
    metrics_ordered = ['loss', 'acc', 'auc_roc']  # the return value from model.evaluate()
    
    @staticmethod
    def get_model_dir(dir_type='nns_model', create_dir=False): 
        # alternatively, use seqConfig.tsHanlder.get_nns_model_dir()
        # params: 
        #   create_dir: if True, then create a new directory if it doesn't exist already
        from tset import TSet 
        from seqConfig import tsHandler

        modelPath = TSet.getPath(cohort=tsHandler.cohort, dir_type=dir_type, create_dir=True)  # ./data/<cohort>/model
        if not os.path.exists(modelPath) and create_dir:
            prefix = os.path.dirname(modelPath) # prefix must already existed
            assert os.path.exists(prefix), "modelEvaluate> Invalid prefix: %s" % prefix
            print('modelEvaluate> Creating model directory: %s ...' % modelPath)
            os.makedirs(modelPath)

        print('get_model_dir> model dir: %s' % modelPath)
        return modelPath

    @staticmethod
    def make_file_id(): # [classifier_name, cohort, d2v_method, seq_ptype, meta]
        from seqConfig import tsHandler
        import seqparams

        identifier = NNS.identifier
        if identifier is None: 
            # cohort_name = tsHandler.cohort
            # d2v_method = tsHandler.d2v  # vector.D2V.d2v_method
            # ctype = tsHandler.ctype
            # clf_name = NNS.model_name
            # identifier = seqparams.makeID(params=[clf_name, cohort_name, d2v_method, ctype, tsHandler.meta])  # null characters and None will not be included
            identifier = 'ep%sb%s' % (NNS.epochs, NNS.batch_size)
        return identifier

    @staticmethod
    def save(model, identifier=None): 
        if identifier is None: identifier = NNS.make_file_id()

        # serialize model to JSON
        model_json = model.to_json()
        path_model = NNS.get_model_dir(dir_type='nns_model', create_dir=True)
        model_fname = 'D%s-%s.json' % (NNS.model_name, identifier)
        fpath = os.path.join(path_model, model_fname)

        with open(fpath, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        # [todo] use LSTM's properties as part of the identifier (e.g. n_units)
        weight_fname = 'D%s-%s.h5' % (NNS.model_name, identifier)
        fpath = os.path.join(path_model, weight_fname)
        
        model.save_weights(fpath)
        print("Saved model to disk at:\n%s\n" % fpath)
        return 

    @staticmethod
    def load(identifier=None): 
        from keras.models import model_from_json

        if identifier is None: identifier = NNS.make_file_id()

        # load json and create model
        path_model = NNS.get_model_dir(dir_type='nns_model', create_dir=True)
        
        # model structure
        model_fname = 'D%s-%s.json' % (NNS.model_name, identifier)
        fpath = os.path.join(path_model, model_fname)

        json_file = open(fpath, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        
        loaded_model = model_from_json(loaded_model_json)

        # model weights 
        weight_fname = 'D%s-%s.h5' % (NNS.model_name, identifier)
        fpath = os.path.join(path_model, weight_fname)

        # load weights into new model
        loaded_model.load_weights(fpath)
        print("Loaded model from disk")

        return loaded_model
    
    @staticmethod
    def summary(): 
        print('NNS> patience: %d, epochs: %d, batch_size: %d' % (NNS.patience, NNS.epochs, NNS.batch_size))

        return

# define roc_callback 
def auc_roc(y_true, y_pred):
    """

    Reference
    ---------
    https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
    """
    # any tensorflow metric
    # tf.metrics.auc
    # value, update_op = tf.metrics.auc(y_pred, y_true)
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def demo_auc_roc(): 
    """

    Memo
    ----
    1. EarlyStopping 

       patience: number of epochs with no improvement after which training will be stopped.

       mode: {'min', 'max', 'auto', }
             In min mode, training will stop when the quantity monitored has stopped decreasing; 
             in max mode it will stop when the quantity monitored has stopped increasing; 
             in auto mode, the direction is automatically inferred from the name of the monitored quantity.
    """
    from keras.utils import np_utils
    from keras.callbacks import Callback, EarlyStopping

    # generation a small dataset
    N_all = 10000
    N_tr = int(0.7 * N_all)
    N_te = N_all - N_tr

    n_features, n_classes = 20, 5
    X, y = make_classification(n_samples=N_all, n_features=n_features, n_classes=n_classes, n_informative=10, random_state=0)
    print('  + dim(X):%s, dim(y):%s' % (str(X.shape), str(y.shape)))
    y = np_utils.to_categorical(y, num_classes=n_classes)  # Converts a class vector (integers) to binary class matrix.

    X_train, X_valid = X[:N_tr, :], X[N_tr:, :]
    y_train, y_valid = y[:N_tr, :], y[N_tr:, :]

    # model & train
    model = Sequential()
    model.add(Dense(n_classes, activation="softmax", input_shape=(X.shape[1],)))

    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy', auc_roc])

    my_callbacks = [EarlyStopping(monitor='auc_roc', patience=300, verbose=1, mode='max'), ]
    model.fit(X, y,
          validation_split=0.3,
          shuffle=True,
          batch_size=32, epochs=10, verbose=1,
          callbacks=my_callbacks)

    # # or use independent valid set
    # model.fit(X_train, y_train,
    #           validation_data=(X_valid, y_valid),
    #           batch_size=32, epochs=10, verbose=1,
    #           callbacks=my_callbacks)
    return model

def make_lstm(n_units=100, n_timesteps=10, n_features=50, n_classes=6, **kargs): 
    """

    Memo
    ----
    n_timesteps = kargs.get('last_n_visits', 10)
    n_features = kargs.get('n_features', vector.D2V.n_features * 2 if tsHandler.d2v in ('pv-dm2', ) else vector.D2V.n_features)

    Examples 
    --------
    model = make_lstm(n_units=1000, n_timesteps=n_timesteps, n_features=n_features, 
                        n_classes=n_classes, dropout_rate=0.2, optimizer=optimizer) 



    Models
    ------
    1. One-to-one: use a Dense layer as you are not processing sequences:
    
       model.add(Dense(output_size, input_shape=input_shape))

    2. One-to-many: this option is not supported well as chaining models is not very easy in Keras so the following version is the easiest one:

       model.add(RepeatVector(number_of_times, input_shape=input_shape))
       model.add(LSTM(output_size, return_sequences=True))

    3. Many-to-one: actually your code snippet is (allmost) example of this approach:
        model = Sequential()
        model.add(LSTM(1, input_shape=(timesteps, data_dim)))

    4. Many-to-many: This is the easiest snippet when length of input and output matches the number of reccurent steps:
        
       model = Sequential()
       model.add(LSTM(1, input_shape=(timesteps, data_dim), return_sequences=True))

       but in general many-to-many when number of steps differ from input/output length: this is hard in Keras

    
                                        O O O
                                        | | |
                                  O O O O O O
                                  | | | | | | 
                                  O O O O O O

        model = Sequential()
        model.add(LSTM(1, input_shape=(timesteps, data_dim), return_sequences=True))
        model.add(Lambda(lambda x: x[:, -N:, :]    // Where N is the number of last steps you want to cover (on image N = 3).

    5. Stacked LSTM (n_layers >=2)
        
        e.g. 

        # expected input data shape: (batch_size, timesteps, data_dim)
        model = Sequential()
        model.add(LSTM(32, return_sequences=True,
                    input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(32))  # return a single vector of dimension 32
        model.add(Dense(10, activation='softmax'))


    Reference 
    ---------
    1. https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/

    """
    def model_summary(): 
        print('make_lstm> n_units=%d, r_dropout=%f, n_layers=%d' % (n_units, kargs.get('dropout_rate', 0.2), kargs.get('n_layers', 1))) 

        msg = ''
        msg += '           optimizer:     %s\n' % kargs.get('optimizer', 'adam')
        msg += '           loss function: %s\n' % kargs.get('loss', 'categorical_crossentropy')
        msg += '           n_timesteps: %d, n_features: %d, n_classes: %d\n\n' % (n_timesteps, n_features, n_classes)
        print msg

    # from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout, Flatten

    n_layers = kargs.get('n_layers', 1)

    # define model
    model = Sequential()
    if n_layers == 1: 
        model.add(LSTM(n_units, input_shape=(n_timesteps, n_features)))  # n_units: number of LSTM units
    else: 
        print('make_lstm> stacked LSTM: n_layers=%d' % n_layers)
        model.add(LSTM(n_units, input_shape=(n_timesteps, n_features), return_sequences=True))
        nl = n_layers-1
        while nl > 0: 
            model.add(LSTM(n_units)) 
            nl -= 1

    # experiment with stacked-LSTM 
    # model.add()
    r_dropout = kargs.get('dropout_rate', 0.2)
    model.add(Dropout(r_dropout))
    model.add(Dense(n_classes, activation='softmax'))
           
    # Compile model
    model.compile(loss=kargs.get('loss', 'categorical_crossentropy'), 
        optimizer=kargs.get('optimizer', 'adam'), metrics=['accuracy', auc_roc])

    model_summary()
    if kargs.get('save_init_weights', True): model.save(NNS.init_weights)

    return model

def gen_lstm(n_units=100, n_timesteps=10, n_features=50, n_classes=6, **kargs):
    def make_lstm(): 
        model = Sequential()
        model.add(LSTM(n_units, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(kargs.get('dropout_rate', 0.2)))
        model.add(Dense(n_classes, activation='softmax'))
        return model 
    from keras.layers import Dense, LSTM, Dropout, Flatten

    n_layers = kargs.get('n_layers', 1)

    return make_lstm

def makeClassifier(X, y, **kargs):
    def eval_weights(y):  # compute class weights and sample weights 
        from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

        # Instantiate the label encoder
        y_uniq = np.unique(y)
        le = preprocessing.LabelEncoder()
        le.fit(y_uniq) # Fit the label encoder to our label series
        print('  + lookup:\n%s\n' % dict(zip(y_uniq, le.transform(y_uniq))) )

        y_int = le.transform(y)   # Create integer based labels Series
        labelToInteger = dict(zip(y, y_int)) # Create dict of labels : integer representation

        class_weight = compute_class_weight('balanced', np.unique(y_int), y_int)
        sample_weight = compute_sample_weight('balanced', y_int)
        class_weight_dict = dict(zip(le.transform(list(le.classes_)), class_weight))

        return class_weight_dict

    # deep learning packages
    from dnn_demo import make_lstm as build_fn_lstm
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout
    from keras.wrappers.scikit_learn import KerasClassifier  ###
    # from keras.utils import np_utils
    # from sklearn.pipeline import Pipeline
    from functools import partial

    y_uniq = np.unique(y)
    n_classes = len(y_uniq)
    assert len(X.shape) >= 3
    n_samples, n_timesteps, n_features = X.shape[0], X.shape[1], X.shape[2]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=kargs.get('test_size', 0.2), 
    #         random_state=kargs.get('random_state', int(time.time())%1000))
    # print('modelEvaluate0> dim(X_train):%s, dim(X_test): %s' % (str(X_train.shape), str(X_test.shape))) # (1651, 10, 50), (709, 10, 50)
    
    class_weight_dict = eval_weights(y) # this has to come before binarzing
    print('makeClassifier> class weights:\n%s\n' % class_weight_dict)

    # vary NNS via this function
    build_fn = kargs.get('build_fn', None)
    if build_fn is None: 
        
        # architecture
        n_layers = kargs.get('n_layers', 1)
        n_units = kargs.get('n_units', 100)

        # regularization
        r_dropout = kargs.get('dropout_rate', 0.2)
        # build_fn = gen_lstm(n_units=100, n_timesteps=n_timesteps, n_features=n_features, n_classes=n_classes) # compiled model

        print('makeClassifier> Using LSTM classifier by default (n_layers:%d, n_units:%d, dropout:%f)' % (n_layers, n_units, r_dropout))

        # modify the model by providing a different model definition in the first argument
        build_fn = partial(build_fn_lstm, n_units=n_units, n_timesteps=n_timesteps, n_features=n_features, n_classes=n_classes, 
                            dropout_rate=r_dropout, n_layers=n_layers)

    my_callbacks = kargs.get('callbacks', None)
    if my_callbacks is None: 
        metric_ms = 'val_loss'  # note: monitor <- 'auc_roc' doesn't seem useful
        mode = 'auto'
        my_callbacks = [EarlyStopping(monitor=metric_ms, min_delta=0, 
            patience=kargs.get('patience', 20), verbose=1, mode=mode)]
        
    model = KerasClassifier(build_fn=build_fn, 

                   # parameters: model.fit(...)
                   validation_split=kargs.get('validation_split', 0.0),  # set to 0 here since we have a separate model evaluation
                   class_weight=class_weight_dict,
                   shuffle=kargs.get('shuffle', True),
                   batch_size=kargs.get('batch_size', NNS.batch_size), epochs=kargs.get('epochs', NNS.epochs),
                   callbacks=my_callbacks, 
                   verbose=1)
    
    return model 

def fit_evaluate(model, data, **kargs):
    """

    Input
    -----
    model: compiled deep NN model
    data: input data dictionary 
          keys: 
             X, y | 
             X_train, y_train, X_test, y_test

    **kargs
    -------
    patience: number of epochs with no improvement after which training will be stopped.
    shuffle

    Memo
    ----
    
    Float between 0 and 1. Fraction of the training data to be used as validation data. 
    The model will set apart this fraction of the training data, will not train on it, 
    and will evaluate the loss and any model metrics on this data at the end of each epoch. 
    The validation data is selected from the last samples in the x and y data provided, before shuffling.
    """
    from keras.utils import np_utils
    from keras.callbacks import Callback, EarlyStopping
    from dnn_utils import auc_roc
    
    my_callbacks = [EarlyStopping(monitor='auc_roc', patience=kargs.get('patience', 300), verbose=1, mode='max')]

    if data.has_key('X') and data.has_key('y'):     
        X, y = data['X'], data['y']
        model.fit(X, y,
              validation_split=kargs.get('validation_split', 0.3),
              shuffle=kargs.get('shuffle', True),
              batch_size=kargs.get('batch_size', 32), epochs=kargs.get('epochs', 200), verbose=1,
              callbacks=my_callbacks) 
    else: 
        X_train, y_train = data['X_train'], data['y_train']
        if data.has_key('X_test') and data.has_key('y_test'): 
            X_test, y_test = data['X_test'], data['y_test']
            model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    batch_size=kargs.get('batch_size', 32), epochs=kargs.get('epochs', 200), verbose=1,
                    callbacks=my_callbacks)
    return model

def load_cohort(**kargs):
    return load_data(**kargs)
def load_data(**kargs): 
    def process_docs(cohort='CKD', ctype='regular', is_simplified=False): 
        ### load + transfomr + (ensure that labeled_seq exists)
        src_dir = kargs.get('inputdir', os.getcwd())
        ifile, ifiles = kargs.get('inputfile', None), kargs.get('ifiles', [])
        if ifile is not None:  # if inputfile is given, it'll take precedence over ifiles input
            ipath = os.path.join(src_dir, ifile)
            assert os.path.exists(ipath), "Invalid input path: %s" % ipath
            ifiles = [ipath, ]  # overwrite ifiles
        print('load_data> inputs:\n%s\n' % ifiles)
        D, L, T = docProc.processDocuments(cohort=cohort, seq_ptype=ctype, 
                    inputdir=src_dir, 
                    ifiles=kargs.get('ifiles', ifiles),  # set to [] to use default
                    # meta=kargs.get('meta', None), 
                    
                    # document-wise filtering 
                    policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  
                    min_ncodes=kargs.pop('min_ncodes', 10),  # retain only documents with at least n codes 

                    # padding? to string? 
                    pad_doc=False, to_str=False, 

                    # content modification
                    # predicate=kargs.get('predicate', None), # reserved for segment_docs()
                    simplify_code=is_simplified, 

                    source_type='default', 
                    create_labeled_docs=False)  # [params] composition

        # Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing
        print('  + nD: %d | cohort=%s, ctype=%s, labeled? %s, simplified? %s' % \
            (len(D), cohort, ctype, is_labeled_data(L), is_simplified))
        return (D, L, T)
    def segment_by_visit(D, L, T, max_length=100, max_n_visits=None, mode='visit'):
        # note that boostrapping used in segmentByVisits is used to preserve the distribution of the codes 
        # but the optimal max_length will depend on the variability and sample size. 

        # [params] 
        docToVisit = docProc.segmentByVisits(D, T, L=L, max_visit_length=max_length) # kargs: max_visit_length
        # docIDs = sorted(docToVisit.keys())  # ascending order

        # D: [ [v1, v2, ... v10], [v1, v2, v3] ... ] v1: [c1 ,c2, c3]

        # kargs: L, n_per_class
        # mode: if 'visit' then each visit is a document 
        #           then one turn visit segments into vectors
        #       if 'uniform' then all visits are combined into a joint document but a max visits specified by last_n_visits
        #           => uniform stands for uniform length: each document is made equal length 
        #       last_n_visits is only effective in mode: uniform 
        #           => limit document length (otherwise, may create too many 'paddings)
        docIDs, D = docProc.visitToDocument(V=docToVisit, last_n_visits=max_n_visits, mode=mode) # each document is a list of visits (a list), in which each visit comprises a list of tokens
        return (docIDs, D) 
    def is_labeled_data(lx): 
        # if lx is None: return False
        nL = len(np.unique(lx)) # len(set(lx))
        if nL <= 1: 
            return False 
        return True
    def verify_docs(D, min_length=10, n_test=100): # args: min_ncodes 
        # 1. minimum length: 'min_ncodes'
        minDocLength = min_length
        if minDocLength is not None: 
            nD = len(D)
            ithdoc = random.sample(range(nD), min(nD, n_test))
            for r in ithdoc: 
                assert len(D[r]) >= minDocLength, "Length(D_%d)=%d < %d" % (r, len(D[r], minDocLength))
        return
    def to_str(D, sep=' ', throw_=False):
        try: 
            D = [sep.join(doc) for doc in D]  # assuming that each element is a string
        except: 
            msg = 'load_data> Warning: Some documents may not contain string-typed tokens!'
            if throw_: 
                raise ValueError, msg 
            else: 
                print msg
            D = [sep.join(str(e) for e in doc) for doc in D]
        return D
    def run_test_case(D, D2):  # given D, L, T 
        nD = len(D)
        assert nD == len(D2)
        for i in random.sample(range(nD), min(nD, 10)): 
            print('... (before):\n%s\n' % D[i])
            print('... (after):\n%s\n' % D2[i])
        return

    import docProc
    
    D, L, T = process_docs(cohort=kargs.get('cohort', 'CKD'), ctype=kargs.get('seq_ptype', 'regular'))
    tSegmentVisit = kargs.get('segment_by_visit', True)

    # max_length: max length for each visit segment after boostrapping
    visitDocIDs, docIDs = [], []
    if tSegmentVisit: 
        mode = kargs.get('mode', 'visit')  # 'visit', 'uniform'
        n_timesteps = 10
        if mode.startswith('v'):  # regular mode that breaks down each document into visit segments 
            # n_timesteps is not used

            # max_n_visit: each document is represented by only the last n visit (but this is ignored in deriving d2v model; it's better to
            #              look at the entire history); this param is relevant when formulating patient-specific vector repr later 
            # max_length: each visit/session is represented by at most this many tokens/codes
            (docIDs, D2) = segment_by_visit(D, L, T, max_length=kargs.get('max_visit_length', 100), max_n_visits=None, mode='visit')
            visitDocIDs = docIDs
            assert len(D2) == len(visitDocIDs) and len(visitDocIDs) >= len(L), \
                      "size(D):%d, size(visitDocIDs): %d, sizes(L): %d" % (len(D2), len(visitDocIDs), len(L))
        else: 
            (docIDs, D2) = segment_by_visit(D, L, T, max_length=kargs.get('max_visit_length', 100), max_n_visits=n_timesteps, mode='uniform')
            if mode.startswith('uniform'): run_test_case(D, D2)
        D = D2
        
    # # from list of list (of tokens) TO list of strings
    # # if padding occurs first, then need to convert back to string-typed tokens
    # if kargs.get('to_str', False): 
    #     D = to_str(D)  # use ' ' by default
    #     # if mode.startswith('uniform'): run_test_case(D, D2)
    #     # D = D2

    # func_segment, p_segment = kargs.get('predicate', None), kargs.get('policy_segment', 'regular')
    # docIDs, D, L, T = segment_docs(D, L, T, predicate=func_segment, policy=p_segment)  # <- policy_segment, inclusive?, include_active?

    ### side effects 
    # if kargs.get('create_encoded_docs', False): 
    #     tsHandler.save_mcs(D, T, L, index=0, docIDs=docIDs, meta=user_file_descriptor, sampledIDs=sampledIDs)  # D has included augmented if provided        

    
    if kargs.get('label_doc', False): 
        if tSegmentVisit: 
            D = labelize(D, label_type='v')  # each "document" is actually just a (bootstrapped) visit segment 
        else: 
            D = labelize(D, label_type='doc')  # each "document" is actually just a (bootstrapped) visit segment   
    
    # organize outputs
    ret = {}
    ret['D'], ret['L'], ret['T'] = D, L, T 
    ret['visitDocIDs'] = visitDocIDs
    ret['docIDs'] = docIDs

    return ret 

def create_encoded_docs(): 

    return

def labelize(docs, class_labels=[], label_type='doc', offset=0): # essentially a wrapper of labeling.labelize 
    import vector
    return vector.labelDocuments(docs, class_labels=class_labels, label_type=label_type, offset=offset) # overwrite_=False

def tokenize(docs=[], filters=None): 
    """

    Memo
    ----
    word_counts: A dictionary of words and their counts.
    word_docs: A dictionary of words and how many documents each appeared in.
    word_index: A dictionary of words and their uniquely assigned integers.
    document_count:An integer count of the total number of documents that were used to fit the Tokenizer.

    """
    from keras.preprocessing.text import Tokenizer
    
    # example documents
    if not docs: 
        docs = ['Well done!',
                'Enjoy life!!',
                'Great effort!!!',
                'Balance between science and spirituality',
                'Word-life balance!', 
                'x y z u', 
                '120.7, 250.0 250.11 toxic scam', ]

    # create the tokenizer
    # filter default (original): filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    # filter default (project): '!"#$%&()*+,/;<=>?@[\]^`{|}~'
    if filters is None: filters = '!"#$%&()*+,/;<=>?@[\]^`{|}~'  # '.', ':', '_' '-' should be consiidered as part of the codes/tokens
    t = Tokenizer(filters=filters)  

    # fit the tokenizer on the documents
    t.fit_on_texts(docs)

    # summarize what was learned
    # print(t.word_counts)
    # print(t.document_count)
    # print(t.word_index)
    # print(t.word_docs)

    return t

def docToInteger(D, policy='exact', **kargs): 
    import docProc 
    return docProc.docToInteger(D, policy=policy, **kargs)

def intDocToDoc(D, tokenizer, lowercase_tokens=['unknown', ], sep=' '):
    index_word = dict(map(reversed, tokenizer.word_index.items()))  # reverse_word_map
    for i, doc in enumerate(D): 
        D[i] = sep.join(index_word[e].upper() for e in doc)
    return D

def padSequences(sequences=[], maxlen=None, value=0.0): 
    """

    Memo
    ----
    1. pad_sequences in Keras does not support strings 
       
       https://stackoverflow.com/questions/46323296/keras-pad-sequences-throwing-invalid-literal-for-int-with-base-10


    """
    from keras.preprocessing.sequence import pad_sequences

    # example sequences
    if not sequences: 
        sequences = [
             [1, 2, 3, 4],
                [1, 2, 3],
                      [1]
        ]
    # pad sequence
    if maxlen is None: 
        # then we need to find out the max length among all documents 
        maxlen = -1 
        for seq in sequences: 
            if len(seq) > maxlen: maxlen = len(seq)

    padded = pad_sequences(sequences, maxlen=maxlen, value=value)  
    print(padded)
    return padded

def embeddig_layer(): 

    # load pre-trained 
    embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
    return

def cnn_classifier_demo1(**kargs): 
    """
    
    Reference
    ---------
    1. https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    """
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
    
    return model
def train_cnn_classifier_demo1(x_train, y_train, x_val, y_val, **kargs):
    model = cnn_classifier_demo1(**kargs)
    # happy learning!
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=2, batch_size=128)
    return model

def t_sentiment_classification(): 
    # CNN for the IMDB problem
    from keras.datasets import imdb
    # from keras.models import Sequential
    # from keras.layers import Dense
    # from keras.layers import Flatten
    from keras.layers.convolutional import Convolution1D
    from keras.layers.convolutional import MaxPooling1D
    from keras.layers.embeddings import Embedding
    from keras.preprocessing import sequence

    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

    # pad dataset to a maximum review length in words
    max_words = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words)

    # create the model
    model = Sequential()
    model.add(Embedding(top_words, 32, input_length=max_words))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=2, batch_size=128, verbose=1)
  
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    return

def plot_word_usage(X=None, y=None): 
    # Load and Plot the IMDB dataset
    from keras.datasets import imdb
    from matplotlib import pyplot
    import numpy

    # X: documents
    if X is None or y is None: 
        # load the dataset
        (X_train, y_train), (X_test, y_test) = imdb.load_data()
        X = numpy.concatenate((X_train, X_test), axis=0)
        y = numpy.concatenate((y_train, y_test), axis=0)

    # summarize size
    print("Training data: ")
    print(X.shape)
    print(y.shape)

    # Summarize number of classes 
    print("Classes: ")
    print(numpy.unique(y))

    # Summarize number of words
    print("Number of words: ")
    print(len(numpy.unique(numpy.hstack(X))))

    # Summarize review length
    print("Review length: ")
    result = map(len, X)
    print("Mean %.2f words (%f)" % (numpy.mean(result), numpy.std(result)))

    # plot review length as a boxplot and histogram
    pyplot.subplot(121)
    pyplot.boxplot(result)
    pyplot.subplot(122)
    pyplot.hist(result)
    pyplot.show()

    return

def t_tokenize(**kargs): 
    """

    Memo
    ----
    1. tokenized output attributes 
       word_counts: A dictionary of words and their counts.
       word_docs: A dictionary of words and how many documents each appeared in.
       word_index: A dictionary of words and their uniquely assigned integers.
       document_count:An integer count of the total number of documents that were used to fit the Tokenizer.

    2. num_words: the maximum number of words to keep, based on word frequency. Only the most common num_words words will be kept.
       filters: a string where each element is a character that will be filtered from the texts. 
                The default is all punctuation, plus tabs and line breaks, minus the ' character.

       lower: boolean. Whether to convert the texts to lowercase.
       split: str. Separator for word splitting.
       char_level: if True, every character will be treated as a token.
       oov_token: if given, it will be added to word_index and used to replace out-of-vocabulary words during text_to_sequence calls

    3. Tokenizer class was written under the assumption that you'll be using it with an Embedding layer, 
    where the Embedding layer expects input_dim to be vocab size + 1. The Tokenizer reserves the 0 index for masking 
    (even though the default for the Embedding layer is mask_zero=False...), so you are only actually tokenizing 
    the top num_words - 1 words.
    """
    def tokenize_n_words(docs, n=None, filters='!"#$%&()*+,/;<=>?@[\]^`{|}~'): 
        # create the tokenizer
        if n is None: 
            t = Tokenizer(filters=filters)
        else: 
            t = Tokenizer(num_words=n+1, filters=filters, oov_token=None)  # n+1 because 0 is a reserved index; see memo [3]
    
        # fit the tokenizer on the documents
        t.fit_on_texts(docs)

        return t

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.text import text_to_word_sequence
    
    # define 5 documents
    docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
           'Excellent work!', 'Summer is warm.', 
           'I am falling in love with Quantum Mechanics.']  # work should have a low tf-idf score

    t = None
    filters = '!"#$%&()*+,/;<=>?@[\]^`{|}~'  # '.', ':', '_' '-' should be consiidered as part of the codes/tokens
    for nw in [None, 5]: 
        if nw is not None: print('> tokenize %d most frequent words only.' % nw)
        t = tokenize_n_words(docs, n=nw, filters=filters)
    
        # summarize what was learned
        # print(t.word_counts)
        # print(t.document_count)
        # print(t.word_index)
        # print(t.word_docs)

        words = sorted([(w, i) for w, i in t.word_index.items()], key=lambda x:x[1], reverse=False)
        print('+ w-coordinate (n=%d):\n%s\n' % (len(words), words))  # e.g. 18 tokens => vec: 18+1=19-D with 0 as a reserved index

    print('-' * 100)     

    # integer encode documents
    for mode in ['count', 'tfidf', ]: 
        print('> encode documents via mode=%s ...' % mode)
        encoded_docs = t.texts_to_matrix(docs, mode=mode)  # first dim == 0
        for i, doc in enumerate(docs): 
            wids = sorted([t.word_index[w] for w in text_to_word_sequence(doc, filters=filters)])
            print('[%d] %s | wid: %s' % (i, doc, wids)) # [t.word_index[w] for w in doc]
            print('     %s' % encoded_docs[i])

    return

def makeTSetDoc(Din, T, L, docIDs, **kargs):
    """
    Create training set directly by treating the whole MCS as a document. 
    This is essentially thes same as makeTSet() by separating document processing operations 
    from the doc2vec model. 
    """
    def get_model_dir(): 
        dir_type = 'model'
        modelPath = TSet.getPath(cohort=tsHandler.cohort, dir_type=dir_type, create_dir=True)  # ./data/<cohort>/model
        print('makeTSetDoc> model dir: %s' % modelPath)
        return modelPath
    def save_tset(X, y, cv_id=0, docIDs=[], meta=None, sparse=False, shuffle_=True): # [params] (X_train, X_test, y_train, y_test)
        return tsHandler.save(X, y, index=cv_id, docIDs=docIDs, meta=meta, sparse=sparse, shuffle_=False)
    def load_tset(cv_id=0, meta=None, sparse=False):  # cohort, seq_ptype, d2v_method (focused_labels, label_map)
        # this should conform to the protocal in makeTSetCombined()
        return tsHandler.load_tset(index=cv_id, meta=meta, sparse=sparse)
    def is_sparse(): 
        if tsHandler.d2v.startswith(('bow', 'bag', 'aphri')):  # aphridate
            return True 
        return False
    import vector 
    from seqConfig import tsHandler
    # from tset import TSet
    
    user_file_descriptor = meta = kargs.get('meta', tsHandler.meta)
    cohort_name = kargs.get('cohort', 'CKD')

    nDoc, nDocEff = len(T), len(Din)
    nTrials = kargs.get('n_trials', 1)
    for cv_id in range(nTrials):   # loop reserved for CV or random subsampling if necessary
        print('    + computing document vectors nD:%d => nDEff: %d ...' % (nDoc, nDocEff))

        # [note] Dl includes augmented if provided: Dl + Dal
        #        this will save a model file
        y = np.array(L)
        X = vector.getDocVec(docs=Din, d2v_method=vector.D2V.d2v_method, 
                                outputdir=get_model_dir(),  # [params] cohort, dir_type='model' 
                                meta=user_file_descriptor,   # user-defined file ID: get_model_id(y=L), # {'U', 'L', 'A'} + ctype + cohort
                                labels=L,  # assess() will evaluate d2v accuracy based on the labels

                                test_=kargs.get('test_model', True), 
                                load_model=kargs.get('load_model', True),

                                segment_by_visit=False, # need to distinguish from 'normal' document 

                                # paramters for sparse repr
                                max_features=kargs.get('max_features', None), # only applies to d2v_method <- 'bow' (bag-of-words sparse model) 
                                text_matrix_mode=kargs.get('text_matrix_mode', 'tfidf'), # only applies to sparse matrix

                                cohort=cohort_name)

        assert X.shape[0] == nDocEff and X.shape[0] == len(L)

        # save training data
        if kargs.get('save_', True):  
            # [test]
            if is_sparse(): print('makeTSetDoc> Sparse matrix (X): dim=%s, d2v=%s' % (str(X.shape), d2v_method))
            ts = save_tset(X, L, cv_id, docIDs=docIDs, meta=user_file_descriptor, sparse=is_sparse()) # [params] X_train, X_test, y_train, y_test
                
        # save the document that produced the training data (based on d2v model)
        if kargs.get('save_doc', False): 
            # user_file_descriptor is tsHandler.meta by default
            tsDoc = save_mcs(D, T, L, index=cv_id, docIDs=docIDs, meta=user_file_descriptor, sampledIDs=sampledIDs)  # D has included augmented if provided
            
            # [test]
            assert X.shape[0] == tsDoc.shape[0], "Size inconsistent: size(doc): %d but size(X): %d" % (tsDoc.shape[0], X.shape[0])

    ### end foreach trial 
    return (X, y) 

def makeTSetVisit(Din, T, L, visitDocIDs, **kargs): 
    """

    Params
    ------
    meta: 
    cohort
    last_n_visits 
    docIDs:       
       if segmented, then docIDs will not be identical to the original document indices. 


    """
    def get_model_dir(): 
        dir_type = 'model'
        modelPath = TSet.getPath(cohort=cohort_name, dir_type=dir_type, create_dir=True)  # ./data/<cohort>/model
        print('makeTSetVisit> model dir: %s' % modelPath)
        return modelPath

    import vector
    from tset import TSet
    from seqConfig import tsHandler

    user_file_descriptor = kargs.get('meta', 'test')
    cohort_name = kargs.get('cohort', 'CKD')
    docIDs = kargs.get('docIDs', [])

    nDoc, nDocEff = len(T), len(Din)
    nTrials = kargs.get('n_trials', 1)
    for cv_id in range(nTrials):   # loop reserved for CV or random subsampling if necessary
        print('    + computing document vectors nD:%d => nDEff: %d ...' % (nDoc, nDocEff))

        # [note] Dl includes augmented if provided: Dl + Dal
        #        this will save a model file
        y = np.array(L)
        X = vector.getDocVec(docs=Din, d2v_method=vector.D2V.d2v_method, 
                                outputdir=get_model_dir(),  # [params] cohort, dir_type='model' 
                                meta=user_file_descriptor,   # user-defined file ID: get_model_id(y=L), # {'U', 'L', 'A'} + ctype + cohort
                                # labels=L,  # assess() will evaluate d2v accuracy based on the labels

                                test_=kargs.get('test_model', True), 
                                load_model=kargs.get('load_model', True),

                                segment_by_visit=True, # need to distinguish from 'normal' document 

                                max_features=None, # only applies to d2v_method <- 'bow' (bag-of-words sparse model) 
                                cohort=cohort_name)

        # Need to combine visit vectors so that each document maps to a vector
        # policy: i) average ii) concategate the last 10 visits
        n_features, n_timesteps = X.shape[1], kargs.get('last_n_visits', 10)  # pv-dm2: vector.D2V.n_features * 2

        # consolidate visit segments and their vectors to form a single document vector for each pateint document set 
        print('check> input prior to consolidateVisits dim(X): %s, len(visitDocIDs): %d' % (str(X.shape), len(visitDocIDs)))  # (20K+, 200)
        X = vector.consolidateVisits(X, y=y, docIDs=visitDocIDs, last_n_visits=n_timesteps)  # flatten out n visit vectors by concatenating them
        assert X.shape[0] == len(y), "Since visit vectors are flattened out i.e. N visit vectors => one big vector, size(X): %d ~ size(y): %d" % \
            (X.shape[0], len(y))

        ### Save training set
        if kargs.get('save_', True):   
            if not docIDs: docIDs = range(len(X))  
            ts = tsHandler.save(X, y, cv_id, docIDs=docIDs, meta=user_file_descriptor, shuffle_=False)
            # save_tset(X, y, cv_id, docIDs=docIDs, meta=user_file_descriptor, shuffle_=False) 

        print('status> Model computation complete (@nTrial=%d)' % cv_id)

    ### end foreach trial 
    return (X, y) 

def modelEvaluateGridSearch(X, y, **kargs): 
    def define_classifier(n_timesteps, n_features, n_classes, **kargs):
        n_layers = kargs.get('n_layers', 1)
        n_units = kargs.get('n_units', 100)
        r_dropout = kargs.get('dropout_rate', 0.2)

        build_fn = kargs.get('build_fn', None)
        if build_fn is None: 
            # build_fn = gen_lstm(n_units=100, n_timesteps=n_timesteps, n_features=n_features, n_classes=n_classes) # compiled model

            print('choose_classifier> using default LSTM classifier (n_layers:%d, n_units:%d, dropout:%f)' % \
                (n_layers, n_units, r_dropout))

            # modify the model by providing a different model definition in the first argument
            build_fn = partial(build_fn_lstm, n_units=n_units, n_timesteps=n_features, n_features=n_features, n_classes=n_classes, 
                                    dropout_rate=r_dropout, n_layers=n_layers)

        # params for model.fit()
        # 1. Callback: 
        #    patience 
        # 2. model.fit() 
        #    shuffle, batch_size, epochs
        my_callbacks = kargs.get('callbacks', None)
        if my_callbacks is None: 
            my_callbacks = [EarlyStopping(monitor='auc_roc', patience=kargs.get('patience', NNS.patience), verbose=1, mode='max')]
        model = KerasClassifier(build_fn=build_fn, 
                   validation_split=0.0,  # set to 0 here since we have a separate model evaluation
                   shuffle=kargs.get('shuffle', True),
                   batch_size=kargs.get('batch_size', NNS.batch_size), epochs=kargs.get('epochs', NNS.epochs),
                   callbacks=my_callbacks, 
                   verbose=1)
    
        return model

    from dnn_demo import make_lstm as build_fn_lstm
    import modelSelect as ms
    # from system.utils import div
    import plotUtils as putils
    from seqConfig import tsHandler
    from seqparams import Graphic
    from sampler import sampling
    from tset import TSet
    from sklearn.model_selection import train_test_split
    from keras.utils import np_utils
    from keras.callbacks import Callback, EarlyStopping
    from sklearn import preprocessing
    import time

    y_uniq = np.unique(y)
    n_classes = len(y_uniq)
    

    # create model
    # X: [n_samples, n_timesteps, n_features]
    assert len(X.shape) >= 3
    n_samples, n_timesteps, n_features = X.shape[0], X.shape[1], X.shape[2]
    
    # [params] architecture 
    n_layers = kargs.get('n_layers', 1)
    n_units = kargs.get('n_units', 100)
    dropout_rate = kargs.get('dropout_rate', 0.2)

    # model selection: {'n_units': [1, 3, 5, 10, 20, 50, 75, 100, ], 'dropout_rate': [0.05, 0.1, 0.2, 0.3, 0.5, ]}
    model = makeClassifier(X, y, build_fn=build_fn_lstm, 
        n_layers=n_layers, n_units=n_units, dropout_rate=dropout_rate, # params for architecture 
        callbacks=None)  # callbacks: None to use default 
    
    # define the grid search parameters
    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 100]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)  # use this to choose batch_size and epochs
    grid_result = grid.fit(X, Y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return

def modelEvaluate0(X, y, model, **kargs): 
    def params_summary(): 
        n_epochs_no_improv_tolerance = kargs.get('patience', 300)
        validation_split_ratio = 0.0

        metric = 'auc_roc'
        batch_size = kargs.get('batch_size', 32)
        epochs = kargs.get('epochs', 200)
        shuffle = kargs.get('shuffle', True)
        kargs.get('epochs', 200)

        print('=' * 100)
        print('... early stopping>')
        print("       + patience: %d, model='max'" % n_epochs_no_improv_tolerance)
        print('... model training>')
        print("       + batch_size: %s, epochs: %s, shuffle? %s, metric: %s" % \
            (batch_size, epochs, shuffle, metric))
        print('=' * 100)
        return 
    def binarize_label(y_uniq):  # dense layer has as many nodes as number of classes 
        lb = preprocessing.LabelBinarizer()  # this converts to binary codes
        class_labels = np.unique(y_uniq)
        lb.fit_transform(class_labels)
        return lb # dict(lookup)
    def general_evaluation(X_test, y_test, trained_model, lb=None):
        if lb is None: lb = binarize_label(y_test)
        y_test = lb.transform(y_test)
        scores = trained_model.evaluate(X_test, y_test, verbose=0)
        print('modelEvaluate0> scores: %s' % scores)  # loss, acc, auc_roc 
        # res = {} # output
        # for i, score in enumerate(scores): 
        #     metric = NNS.score_map[i]
        #     res[metric] = score   # use this to assess over/under fitting i.e gap between training and test errors
        return scores
    def history_evluation(history, metrics_ordered=None, n_metrics=None):
        # available metrics: 
        # ['acc', 'loss', 'val_auc_roc', 'auc_roc', 'val_acc', 'val_loss'] i.e. metric M, and val_M where M = acc, loss, ... 
        metrics = [k for k in history.history.keys() if not k.startswith('val')]
        if n_metrics is None: n_metrics = len(metrics)
        # scores 
        score_map = {0:'loss', 1:'acc', 2:'auc_roc'}
        
        lastn = 10
        scores = {}
        gaps = {}
        for mt in metrics: 
            mv = 'val_%s' % mt
            score = np.mean(history.history[mt][-lastn:])
            val_score = np.mean(history.history[mv][-lastn:])
            gaps[mt] = abs(score-val_score)
            scores[mt] = score

        # output format? 
        # dictionaries to vectors
        #    => conform to the order of model.evaluate()  # loss, accuracy, auc_roc 
        # [hardcode]
        if metrics_ordered is None: metrics_ordered = NNS.metrics_ordered  # ('loss', 'acc', 'auc_roc', )
        scores = [scores[mt] for mt in metrics_ordered] 
        gaps = [gaps[mt] for mt in metrics_ordered] 

        return (scores, gaps)
    def eval_weights(y):  # compute class weights and sample weights 
        from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

        # Instantiate the label encoder
        y_uniq = np.unique(y)
        le = preprocessing.LabelEncoder()
        le.fit(y_uniq) # Fit the label encoder to our label series
        print('  + lookup:\n%s\n' % dict(zip(y_uniq, le.transform(y_uniq))) )

        y_int = le.transform(y)   # Create integer based labels Series
        labelToInteger = dict(zip(y, y_int)) # Create dict of labels : integer representation

        class_weight = compute_class_weight('balanced', np.unique(y_int), y_int)
        sample_weight = compute_sample_weight('balanced', y_int)
        class_weight_dict = dict(zip(le.transform(list(le.classes_)), class_weight))

        return class_weight_dict

    def train_dev_test_split(X, y, ratios=[0.8, 0.1, 0.1, ]): 
        assert len(ratios) in (2, 3, )

        n0 = X.shape[0]
        ridx = sampling.splitDataPerClass(y, ratios=ratios)  # works by permutating data index via y
        assert len(ridx) == 3

        X, y = X[ridx[0]], y[ridx[0]]
        X_dev, y_dev = X[ridx[1]], y[ridx[0]]
        X_test, y_test = X[ridx[2]], y[ridx[0]]

        n1 = X.shape[0]
        print('split> sample size %d (r=%s)=> (train: %d, dev: %d, test: %d)' % (n0, ratios, n1, X_dev.shape[0], X_test.shape[0]))
        assert X.shape[0]+X_dev.shape[0]+X_test[0] == X.shape[0]

        return [(X, y), (X_dev, y_dev), (X_test, y_test)]
    def plot_metric(train='acc', test='val_acc'): # params: n_classes
        # params: m_train: {'acc', 'loss'}
        #         m_val: {'val_acc', 'val_loss'}
        plt.clf()
        plt.plot(history.history[train])
        plt.plot(history.history[test])

        title_msg = kargs.get('title', 'Model Evaluation at Model Selection (n_classes=%d)' % n_classes)
        plt.title(title_msg)

        adict = {'acc': 'accuracy', 'loss': 'loss'}
        plt.ylabel(adict[train])
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        return plt
    def save_plot(plt, identifier=None, metric='acc'): 
        # identifier = NNS.make_file_id()
        params_id = 'ep%sb%s' % (NNS.epochs, NNS.batch_size)
        if not identifier: 
            identifier = params_id
        else: 
            identifier = "%s-%s" % (identifier, params_id)

        # outputfile = kargs.get('outputfile', None)
        outputfile = '%s_eval-%s.tif' % (NNS.model_name, identifier)
        outputdir = kargs.get('outputdir', None)
        if outputdir is None: outputdir = Graphic.getPath(cohort=kargs.get('cohort', tsHandler.cohort), dir_type='plot', create_dir=True)

        # save plot  
        putils.saveFig(plt, fpath=os.path.join(outputdir, outputfile))
        return
    def eval_mode(metric_monitored): 
        mode = 'max'
        if metric_monitored in ('val_loss','loss', ): 
            mode = 'min'
        return mode

    import modelSelect as ms
    # from system.utils import div
    import plotUtils as putils
    from seqConfig import tsHandler
    from seqparams import Graphic
    from sampler import sampling
    from tset import TSet
    from sklearn.model_selection import train_test_split
    from keras.utils import np_utils
    from keras.callbacks import Callback, EarlyStopping
    from sklearn import preprocessing
    import time

    y_uniq = np.unique(y)
    n_classes = len(y_uniq)
    assert len(X.shape) >= 3
    n_samples, n_timesteps, n_features = X.shape[0], X.shape[1], X.shape[2]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=kargs.get('test_size', 0.2), 
    #         random_state=kargs.get('random_state', int(time.time())%1000))
    # print('modelEvaluate0> dim(X_train):%s, dim(X_test): %s' % (str(X_train.shape), str(X_test.shape))) # (1651, 10, 50), (709, 10, 50)
    
    class_weight_dict = eval_weights(y) # this has to come before binarzing
    print('... class weights:\n%s\n' % class_weight_dict)

    # binary encode labels
    lb = binarize_label(y_uniq)

    # early stopping
    params_summary()
    metric_ms = 'val_loss'  # note: monitor <- 'auc_roc' doesn't seem useful
    mode = eval_mode(metric_ms)
    my_callbacks = [EarlyStopping(monitor=metric_ms, min_delta=0, 
        patience=kargs.get('patience', 20), verbose=1, mode=mode)]

    # y_train = lb.transform(y_train)
    y = lb.transform(y)

    # other params: validation_data=(X_test, y_test)
    history = model.fit(X, y,
                validation_split=kargs.get('validation_split', 0.3),  # set to 0 here since we have a separate model evaluation
                class_weight=class_weight_dict, 
                shuffle=kargs.get('shuffle', True),
                batch_size=kargs.get('batch_size', 16), epochs=kargs.get('epochs', 200), verbose=1,
                callbacks=my_callbacks) 

    # ['acc', 'loss', 'val_auc_roc', 'auc_roc', 'val_acc', 'val_loss'] i.e. metric M, and val_M where M = acc, loss, ... 
    print('... history of metrics comprising: %s' % history.history.keys())

    # summarize history for accuracy
    file_id = kargs.get('identifier', '')
    nlastm = 15
    for mt, mv in [('acc', 'val_acc'), ('loss', 'val_loss'), ]: 
        # [test]
        for m in (mt, mv): 
            print('... ... metric (%s): %s (n=%d)' % (m, history.history[m][-nlastm:], len(history.history[m])))
        
        # generalization error 
        # print('... ... metric (gap): %s' % history.history[mt][-nlastm:]-history.history[mv][-nlastm:])
        save_plot(plot_metric(train=mt, test=mv), identifier=file_id, metric=mt)

    scores, gaps = history_evluation(history, metrics_ordered=NNS.metrics_ordered)  # ['loss','acc', 'auc_roc']
    # scores = general_evaluation(X_test, y_test, trained_model=model, lb=lb)  # loss, accuracy, auc

    return (scores, gaps)  # gaps measure the differece between training and validation/test error

def analyze_performance(scores, n_resampled=100):
    from sampler import sampling
    # import collections
    finalRes = {}
    for mt, values in scores.items():  # foreach metric and its performance scores (over multiple trials)

        # min, max computed wrt AUC scores
        if mt in ('min', 'max'):   # stage, score
            # find out majoriy
            counter = collections.Counter([label for label, _ in values])
            print('analyze_perf> %s => \n%s\n' % (mt, counter))
            target = counter.most_common(1)[0][0]
            
            target_values = [s for label, s in values if label == target]
            assert len(target_values) > 0, "No scores found for label: %s" % target
            bootstrapped = sampling.bootstrap_resample(target_values, n=n_resampled)
            finalRes[mt] = (target, np.mean(bootstrapped))

            ret = sampling.ci4(bootstrapped, low=0.05, high=0.95)  # keys: ci_low, ci_high, mean, median, se/error
            finalRes['%s_err' % mt] = (ret['ci_low'], ret['ci_high'])
        else: 

            bootstrapped = sampling.bootstrap_resample(values, n=n_resampled)
            finalRes[mt] = np.mean(bootstrapped)

            ret = sampling.ci4(bootstrapped, low=0.05, high=0.95)  # keys: ci_low, ci_high, mean, median, se/error
            finalRes['%s_err' % mt] = (ret['ci_low'], ret['ci_high']) 
    return finalRes # keys: X in {min, max, micro, macro, loss, acc, auc_roc} and X_err 

def subset(X, y, train_subset=1.0, **kargs):
    # import time
    # from sklearn.model_selection import train_test_split

    # check sample size 
    if train_subset > 1 and train_subset > X.shape[0]:  
        print("modelEvaluateBatch> Warning: sample size=%d < n_subset %d" % (X.shape[0], train_subset))
        return (X, y)  # noop

    # for large dataset, may want to subsample the training data 
    X_subset, _, y_subset, _ = train_test_split(X, y, 
        train_size=train_subset,   # e.g. ratio, say 0.7 or absolute number, say 1000  
        # test_size=kargs.get('test_size', 0.3),   # det_test_size(), # accept test_size, ratios
        random_state=kargs.get('random_state', int(time.time())%1000), stratify=y)
    return (X_subset, y_subset)

def modelEvaluateBatch(X, y, model, **kargs): 
    from sampler import sampling
    # from sklearn.model_selection import train_test_split
    # import time

    nTrials = kargs.get('n_trials', 1)
    targetMetric = kargs.get('target_metric', 'loss')

    ref_performance = 0.5
    scores, finalRes = {}, {}

    tInitPerTrial = kargs.get('reinit_weights', True)
    # train_subset = kargs.get('train_subset', 10000)
    # tSubset = train_subset < 1.0 or train_subset >= 2

    # customize modelEvaluate
    kargs['conditional_save'] = True  # if True, save only the model that performs better than the previous one
    kargs['target_metric'] = targetMetric
    kargs['ref_performance'] = ref_performance

    Xs, ys = X, y
    for i in range(nTrials): 
        # if tSubset: 
        #     Xs, ys = subset(X, y, train_subset=train_subset)
        #     print('modelEvaluateBatch> trial #%d | N=%d' % (i+1, len(Xs)))

        # **kargs: ratio_train, if specified, only use this fraction of data to train model
        #          train_size, test_size
        res = modelEvaluate(X, y, model, **kargs)  # a dictionary with keys: {min, max, micro, macro, loss, acc, auc_roc} 
        
        # update reference preformance: don't save a model unless performance gets better
        ref_performance = res[targetMetric]
        for mt, score in res.items(): 
            if not scores.has_key(mt): scores[mt] = []
            scores[mt].append(score)
        
        if tInitPerTrial: 
            model = reinitialize_weights(model, from_file=NNS.init_weights)

    n_resampled = 100
    res = analyze_performance(scores, n_resampled=n_resampled)

    return res 

def modelEvaluateClassic(X, y, model, **kargs): 
    import modelSelect as ms
    return ms.modelEvaluate(X, y, model, **kargs)

def modelEvaluate(X, y, model, **kargs):
    """
    
    Input
    -----
    classifier: untrained classifier (to fit the data in order to compute optimal weights)

    Memo
    ----
    1. EarlyStopping 

       patience: number of epochs with no improvement after which training will be stopped.

       mode: {'min', 'max', 'auto', }
             In min mode, training will stop when the quantity monitored has stopped decreasing; 
             in max mode it will stop when the quantity monitored has stopped increasing; 
             in auto mode, the direction is automatically inferred from the name of the monitored quantity.

    2. patience argument represents the number of epochs before stopping once your loss starts to increase (stops improving). 
       This depends on your implementation, if you use very small batches or a large learning rate your loss zig-zag 
       (accuracy will be more noisy) so better set a large patience argument. If you use large batches and a small learning rate 
       your loss will be smoother so you can use a smaller patience argument. 


    """
    def roc_evaluation(X_test, y_test, trained_model): # useful for deep learning model, which takes much longer to train
        identifier = make_file_id()
        outputfile = kargs.get('outputfile', None)
        if outputfile is None: outputfile = 'roc-multiclass-%s.tif' % identifier
        outputdir = kargs.get('outputdir', None)
        if outputdir is None: outputdir = Graphic.getPath(cohort=kargs.get('cohort', tsHandler.cohort), dir_type='plot', create_dir=True)

        target_labels = kargs.get('roc_per_class', ['CKD Stage 1', 'CKD Stage 3', 'CKD Stage 5', ])

        # yt: input labels should not be binarized
        # other params: binarize_label (if True, then labels y are binarized prior to fit())
        res = ms.runROCMulticlass(X_test, y_test, trained_model=trained_model, 
            outputdir=outputdir, 
            # outputfile=outputfile,  # use identifier provided but each performance routine will have its own prefix for file naming
            identifier=identifier, 
            target_labels=target_labels) 
        return res 
    def general_evaluation(X_test, y_test, trained_model, lb=None, metrics_ordered=None):
        if lb is None: lb = binarize_label(y_test)
        if metrics_ordered is None: metrics_ordered = NNS.metrics_ordered # ['loss', 'acc', 'auc_roc']
        y_test = lb.transform(y_test)
        scores = trained_model.evaluate(X_test, y_test, verbose=0)
        print('modelEvaluate> scores:\n%s\n' % zip(NNS.metrics_ordered, scores))  # loss, accuracy, auc 
        res = {} # output
        for i, metric in enumerate(metrics_ordered): 
            # metric = NNS.score_map[i]
            res[metric] = scores[i]   # use this to assess over/under fitting i.e gap between training and test errors
        return res
    def history_evluation(history, metrics_ordered=None, n_metrics=None):
        # available metrics: 
        # ['acc', 'loss', 'val_auc_roc', 'auc_roc', 'val_acc', 'val_loss'] i.e. metric M, and val_M where M = acc, loss, ... 
        metrics = [k for k in history.history.keys() if not k.startswith('val')]
        if n_metrics is None: n_metrics = len(metrics)
        # scores 
        # score_map = {0:'loss', 1:'acc', 2:'auc_roc'}
        
        lastn = 10  # average the last n score
        scores = {}
        gaps = {}
        for mt in metrics: 
            mv = 'val_%s' % mt
            score = np.mean(history.history[mt][-lastn:])
            val_score = np.mean(history.history[mv][-lastn:])
            gaps[mt] = abs(score-val_score)
            scores[mt] = score

        # conform to the order of model.evaluate()  # loss, accuracy, auc_roc 
        # [hardcode]
        if metrics_ordered is None: metrics_ordered = NNS.metrics_ordered # ('loss', 'acc', 'auc_roc', )
        scores = [scores[mt] for mt in metrics_ordered] 
        gaps = [gaps[mt] for mt in metrics_ordered] 

        return (scores, gaps)

    def make_file_id(): # [classifier_name, cohort, d2v_method, seq_ptype, meta]
        identifier = kargs.get('identifier', None)
        if identifier is None: 
            cohort_name = tsHandler.cohort
            d2v_method = tsHandler.d2v  # vector.D2V.d2v_method
            ctype = tsHandler.ctype
            clf_name = get_clf_name() #kargs.get('classifier_name', 'nns')
            identifier = seqparams.makeID(params=[clf_name, cohort_name, d2v_method, ctype, 
                            kargs.get('meta', tsHandler.meta)])  # null characters and None will not be included
        return identifier
    def get_clf_name(): # can be called before or after model selection (<- select_classifier)
        name = 'nns'
        try: 
            name = model.__name__
        except: 
            print('info> infer classifier name from class name ...')
            # name = str(estimator).split('(')[0]
            name = model.__class__.__name__
        return name
    def train_dev_test_split(X, y, ratios=[0.8, 0.1, 0.1, ]): 
        assert len(ratios) in (2, 3, )

        n0 = X.shape[0]
        ridx = sampling.splitDataPerClass(y, ratios=ratios)  # works by permutating data index via y
        assert len(ridx) == 3

        X, y = X[ridx[0]], y[ridx[0]]
        X_dev, y_dev = X[ridx[1]], y[ridx[0]]
        X_test, y_test = X[ridx[2]], y[ridx[0]]

        n1 = X.shape[0]
        print('split> sample size %d (r=%s)=> (train: %d, dev: %d, test: %d)' % (n0, ratios, n1, X_dev.shape[0], X_test.shape[0]))
        assert X.shape[0]+X_dev.shape[0]+X_test[0] == X.shape[0]

        return [(X, y), (X_dev, y_dev), (X_test, y_test)]
    def describe_classifier(clf):
        myName = '?'
        try: 
            myName = clf.__class__.__name__ 
        except: 
            print('warning> Could not access __class__.__name__ ...')
            myName = str(clf)
        print('... classifier name: %s' % myName)
        return
    def det_test_size(): # used for train_test_split (without dev set)
        r = 0.3
        if kargs.has_key('test_size'): 
            r = kargs['test_size'] # ratio or absolute size
        else: 
            ratios = kargs.get('ratios', [0.7, ])  # training set ratio 
            r = 1-sum(ratios)
            assert r > 0.0, "test_size ratio < 0.0: %f" % r
        print('runROCMulticlass> test set ratio: %f' % r)
        return r
    def params_summary(): 
        n_epochs_no_improv_tolerance = kargs.get('patience', 300)
        validation_split_ratio = 0.0

        metric = 'auc_roc'
        batch_size = kargs.get('batch_size', 32)
        epochs = kargs.get('epochs', 200)
        shuffle = kargs.get('shuffle', True)
        kargs.get('epochs', 200)

        print('=' * 100)
        print('... early stopping>')
        print("       + patience: %d, model='max'" % n_epochs_no_improv_tolerance)
        print('... model training>')
        print("       + batch_size: %s, epochs: %s, shuffle? %s, metric: %s" % \
            (batch_size, epochs, shuffle, metric))
        print('=' * 100)
        return 
    def binarize_label(y_uniq):  # dense layer has as many nodes as number of classes 
        lb = preprocessing.LabelBinarizer()  # this converts to binary codes
        class_labels = np.unique(y_uniq)
        lb.fit_transform(class_labels)
        # print('binarizer> fit %d labels' % len(class_labels))
        return lb # dict(lookup)
    def save(model):
        # identifier = kargs.get('identifier', None) 
        # if identifier is None: identifier = make_file_id()
        NNS.save(model, identifier=kargs.get('identifier', None)) # params: model_name
        return
    def load(): 
        # identifier = kargs.get('identifier', None) 
        # if identifier is None: identifier = make_file_id()
        return NNS.load(identifier=kargs.get('identifier', None))
    def eval_weights(y):  # compute class weights and sample weights 
        from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

        # Instantiate the label encoder
        y_uniq = np.unique(y)
        le = preprocessing.LabelEncoder()
        le.fit(y_uniq) # Fit the label encoder to our label series
        print('  + lookup:\n%s\n' % dict(zip(y_uniq, le.transform(y_uniq))) )

        y_int = le.transform(y)   # Create integer based labels Series
        labelToInteger = dict(zip(y, y_int)) # Create dict of labels : integer representation

        class_weight = compute_class_weight('balanced', np.unique(y_int), y_int)
        sample_weight = compute_sample_weight('balanced', y_int)
        class_weight_dict = dict(zip(le.transform(list(le.classes_)), class_weight))

        return class_weight_dict
    def eval_mode(metric_monitored): 
        mode = 'max'
        if metric_monitored in ('val_loss','loss', ): 
            mode = 'min'
        return mode
    def save_plot(plt, identifier=None, metric='acc'): 
        # identifier = NNS.make_file_id()
        params_id = 'ep%sb%s' % (NNS.epochs, NNS.batch_size)
        if not identifier: 
            identifier = params_id
        else: 
            identifier = "%s-%s" % (identifier, params_id)

        # outputfile = kargs.get('outputfile', None)
        outputfile = '%s_eval-%s.tif' % (NNS.model_name, identifier)
        outputdir = kargs.get('outputdir', None)
        if outputdir is None: outputdir = Graphic.getPath(cohort=kargs.get('cohort', tsHandler.cohort), dir_type='plot', create_dir=True)

        # save plot  
        putils.saveFig(plt, fpath=os.path.join(outputdir, outputfile))
        return
    def plot_metric(train='acc', test='val_acc'): # params: n_classes
        # params: m_train: {'acc', 'loss'}
        #         m_val: {'val_acc', 'val_loss'}
        plt.clf()
        plt.plot(history.history[train])
        plt.plot(history.history[test])

        title_msg = kargs.get('title', 'Model Evaluation at Model Selection (n_classes=%d)' % n_classes)
        plt.title(title_msg)

        adict = {'acc': 'accuracy', 'loss': 'loss'}
        plt.ylabel(adict[train])
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        return plt

    import modelSelect as ms
    # from system.utils import div
    from seqConfig import tsHandler
    from sampler import sampling
    from seqparams import Graphic
    from tset import TSet

    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    # from sklearn.preprocessing import LabelEncoder

    from keras.utils import np_utils
    from keras.callbacks import Callback, EarlyStopping
    from keras.models import model_from_json
    # import time
    
    N = X.shape[0]
    ratio = kargs.get('ratio', 0.7)  # ratio of data used for training
    train_subset = kargs.get('train_subset', int(N*ratio))
    test_subset = kargs.get('test_subset', N-train_subset)  

    y_uniq = np.unique(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        train_size=train_subset,  # training set (including validation set for model selection) 
        test_size=test_subset,   # det_test_size(), # accept test_size, ratios
        random_state=kargs.get('random_state', int(time.time())%1000), stratify=y)
    print('modelEvaluate> dim(X_train):%s, dim(X_test): %s' % (str(X_train.shape), str(X_test.shape))) # (1651, 10, 50), (709, 10, 50)a
    print('               dim(y_train):%s, dim(y_test):%s' % (str(y_train.shape), str(y_test.shape)))

    # if model is None: model = load() # params: model_name, identifier 

    class_weight_dict = eval_weights(y) # this has to come before binarzing
    print('... class weights:\n%s\n' % class_weight_dict)

    # early stopping
    params_summary()
    target_metric = 'val_loss'  # note: monitor <- 'auc_roc' doesn't seem useful
    mode = eval_mode(target_metric)
    my_callbacks = [EarlyStopping(monitor=target_metric, min_delta=0, 
        patience=kargs.get('patience', 20), verbose=1, mode=mode)]

    # binary encode labels
    lb = binarize_label(y_uniq)
    y_train = lb.transform(y_train)
    ratio_validation = kargs.get('ratio_validation', 0.3)
    history = model.fit(X_train, y_train,
                  validation_split=ratio_validation,  # set to 0 here since we have a separate model evaluation from train_test_split
                  class_weight=class_weight_dict, 
                  shuffle=kargs.get('shuffle', True),
                  batch_size=kargs.get('batch_size', 16), epochs=kargs.get('epochs', 200), verbose=1,
                  callbacks=my_callbacks) 

    print('... history of metrics comprising: %s' % history.history.keys())
    print('... model fitting complete > starting model evaluation')

    # provide the trained model so that the following routine only need to produce label predictions (y_pred/y_score) from which 
    # ROC plot is computed. 

    # summarize history for accuracy
    file_id = kargs.get('identifier', '')
    nlastm = 15
    for mt, mv in [('acc', 'val_acc'), ('loss', 'val_loss'), ]: 
        # [test]
        for m in (mt, mv): 
            print('... ... metric (%s): %s (n=%d)' % (m, history.history[m][-nlastm:], len(history.history[m])))
        
        # generalization error 
        # print('... ... metric (gap): %s' % history.history[mt][-nlastm:]-history.history[mv][-nlastm:])
        save_plot(plot_metric(train=mt, test=mv), identifier=file_id, metric=mt)
    scores, gaps = history_evluation(history, metrics_ordered=NNS.metrics_ordered)  # ['loss','acc', 'auc_roc']

    # y_test = lb.transform(y_test)

    # [note] metrics_ordered should conform to Kera's return value of model.evaluate
    res = general_evaluation(X_test, y_test, trained_model=model, lb=lb)  # params: metrics_ordered: ['loss', 'acc', 'auc_roc', ] by default
    res.update(roc_evaluation(X_test, y_test, trained_model=model))  # attributes: {min, max, micro, macro, loss, accuracy}
    # evaluation 
    # res['micro'] to get micro-averaged AUC

    ### model persistence 
    ref_performance = kargs.get('ref_performance', 0.5)  # only used in modelEvluateBatch()
    if kargs.get('save_model', True): 
        tSaveEff = False
        if kargs.get('conditional_save', False): 
            target_metric = kargs.get('target_metric', 'loss') # this is NOT the same as the monintored metric above
            if target_metric.startswith('los'): 
                if res['loss'] <= ref_performance: tSaveEff = True
            else: 
                if res[target_metric] >= ref_performance: tSaveEff = True
        else: 
            tSaveEff = True

        if tSaveEff: 
            print('... saving a model with metric(%s): %f' % (target_metric, res[target_metric]))
            save(model)

    return res  # a dictionary with keys: {min, max, micro, macro, loss, accuracy, auc}

def multiClassEvaluate(**kargs):  
    import seqClassify as sc 
    return sc.multiClassEvaluate(**kargs)

def multiClassEvaluate0(X, y, classifier, **kargs): 
    def roc_cv(X, y, classifier, fpath=None, target_labels=[], n_folds=5):    
        identifier = make_file_id()
        outputdir = Graphic.getPath(cohort=tsHandler.cohort, dir_type='plot', create_dir=True)
        # classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

        # other params: plot_selected_classes/False
        res = ms.runCVROCMulticlass(X, y, classifier=classifier, 
            prefix=outputdir, identifier=identifier, target_labels=target_labels, 
            n_folds=n_folds,

            general_evaluation=True, # set to True to include loss, accuracy, etc.
            plt_style='seaborn', 
            plot_selected_classes=False) # set to True to avoid cluttering; if False, target_labels will be ignored 
        return res # resturn max performance and min performance; key: 'min': label -> auc, 'max': label -> auc
    def make_file_id(): # [classifier_name, cohort, d2v_method, seq_ptype, meta]
        identifier = kargs.get('identifier', None)
        if identifier is None: 
            cohort_name = tsHandler.cohort
            d2v_method = tsHandler.d2v  # vector.D2V.d2v_method
            ctype = tsHandler.ctype
            clf_name = get_clf_name() #kargs.get('classifier_name', 'nns')
            identifier = seqparams.makeID(params=[clf_name, cohort_name, d2v_method, ctype, 
                            kargs.get('meta', tsHandler.meta)])  # null characters and None will not be included
        return identifier
    def validate_labels(): # targets, y
        assert set(targets).issubset(np.unique(y)), "targets contain unknown labels:\n%s\n" % targets
        return
    def get_clf_name(): # can be called before or after model selection (<- select_classifier)
        # classifier = kargs.get('classifier', None)
        
        if classifier is not None: 
            try: 
                name = classifier.__name__
            except: 
                print('info> infer classifier name from class name ...')
                # name = str(estimator).split('(')[0]
                name = classifier.__class__.__name__
        else: 
            name = kargs.get('classifier_name', None) 
            assert name is not None
        return name 

    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from scipy import interp

    # multiclass evaluation
    from sklearn.metrics import cohen_kappa_score
    from sklearn.metrics import confusion_matrix
    import modelSelect as ms
    from seqparams import Graphic
    from tset import TSet
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score

    # roc_per_class: To prevent the figure from being clobbered, select only, say 3, classes to present in the ROC curve
    targets = kargs.get('roc_per_class', ['CKD Stage 1', 'CKD Stage 3', 'CKD Stage 5', ])  # assuming that CKD Stage 3a, 3b are merged
    validate_labels()
    res = roc_cv(X, y, classifier=classifier, target_labels=targets)  

    return res

def t_classify(**kargs):
    def validate_classes(no_throw=True): 
        n_classes = np.unique(y_train).shape[0]
        n_classes_test = np.unique(y_test).shape[0]
        print('t_classify> n_classes: %d =?= n_classes_test: %d' % (n_classes, n_classes_test))
        return n_classes
    def transform_label(l, positive=None):
        if positive is None: positive = positive_classes # go to the outer environment to find its def
        y_ = labeling.binarize(l, positive=positive) # [params] negative
        return y_ 
    def relabel(): 
        # CKD stage classes
        lmap = {}
        lmap['Control'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
        lmap['CKD Stage 5'] = ['ESRD after transplant', 'ESRD on dialysis', ]
        lmap['CKD Stage 3'] = ['CKD Stage 3a', 'CKD Stage 3b', ]
        print('  + Relabeling the data set according to the following map:\n%s\n' % lmap)
        return lmap
    def subsample(X, y, n=None, sort_index=False, random_state=53):
        """
        Subsampling operation applied after doc vec X is derived. 
        """
        if n is not None: 
            idx = cutils.samplePerClass(y, n_per_class=n, sort_index=sort_index, random_state=random_state)
            print('  + X (dim0: %s -> dim: %s)' % (X.shape[0], len(idx)))
            return (X[idx], y[idx])
        return (X, y)
    def subsample2(ts, n=None, sort_index=True, random_state=53):
        if n is not None: # 0 or None => noop
            ts = cutils.sampleDataframe(ts, col=TSet.target_field, n_per_class=n, random_state=random_state)
            n_classes_prime = check_tset(ts)
            print('  + after subsampling, size(ts)=%d, n_classes=%d (same?)' % (ts.shape[0], n_classes_prime))
        else: 
            # noop 
            pass 
        return ts
    def check_tset(ts): 
        target_field = TSet.target_field
        n_classes = 1 
        no_throw = kargs.get('no_throw', True)
        if ts is not None and not ts.empty: 
            # target_field in ts.columns # o.w. assuming that only 1 class 
            n_classes = len(ts[target_field].unique())  
            print('t_classify> number of classes: %d' % n_classes)
        else:
            msg = 't_classify> Warning: No data found (cohort=%s)' % kargs.get('cohort', 'CKD')
            if no_throw: 
                print msg 
            else: 
                raise ValueError, msg
    def summary(ts=None, X=None, y=None): 
        # experimental_settings
        msg = "... cohort=%s, ctype=%s, d2v=%s, meta=%s\n" % \
                (tsHandler.cohort, tsHandler.ctype, tsHandler.d2v_method, tsHandler.meta)
        msg += "  + is_simplified? %s, ... \n" % tsHandler.is_simplified
        msg += '  + classification mode: %s\n' % mode
        msg += '  + classifiers:\n%s\n' % clf_list
        msg += '  + training set type:%s\n' % ts_dtype

        nrow = n_classes = -1
        if ts is not None: 
            nrow = ts.shape[0]
            n_classes = len(ts[TSet.target_field].unique())
        else: 
            assert X is not None
            nrow = X.shape[0]
            n_classes = len(np.unique(y)) if y is not None else 1
        msg += '  + training set dim:%d\n' % nrow 
        msg += '  + n classes:%d\n' % n_classes

        print msg 
        return 
    def choose_classifier(name='random_forest'):
        if name.startswith(('rand', 'rf')):  # n=389K
            # max_features: The number of features to consider when looking for the best split; sqrt by default
            # sample_leaf_options = [1,5,10,50,100,200,500] for tuning minimum sample leave (>= 50 usu better)
            clf = RandomForestClassifier(n_jobs=10, random_state=53, 
                    n_estimators=100,   # default: 10
                    min_samples_split=250, min_samples_leaf=50)  # n_estimators=500
        elif name.startswith('log'):
            clf = LogisticRegression(class_weight='balanced', solver='saga', penalty='l2')
        elif name.startswith( ('stoc', 'sgd')):  
            clf = SGDClassifier(loss='log', penalty='elasticnet', max_iter=1500, tol=1e-3, n_jobs=10) # l1_ratio: elastic net mixing param: 0.15 by default
        elif name.startswith(('grad', 'gb')):  # gradient boost tree
            # min_samples_split=250, min_samples_leaf=50 
            # max_leaf_nodes: If None then unlimited number of leaf nodes.
            # subsample: fraction of samples to be used for fitting the individual base learners
            #            Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=53, 
                min_samples_split=250, min_samples_leaf=50, max_depth=8,  # prevent overfitting
                max_features = 'sqrt', # Its a general thumb-rule to start with square root.
                subsample=0.85)
        else: 
            raise ValueError, "Unrecognized classifier: %s" % name
        return clf
    def experimental_settings(): 
        print('t_classify> tset type %s, maxNPerClass: %s, drop control? %s' % \
            (kargs.get('tset_dtype', 'dense'), kargs.get('n_per_class', None), kargs.get('drop_ctrl', False)))

        print('  + cohort: %s, ctype: %s' % (tsHandler.cohort, tsHandler.ctype))
        print('  + d2v: %s, params: ' % (tsHandler.d2v, ))
        print('  + userFileID: %s' % kargs.get('meta', tsHandler.meta))

        try: 
            print('  + using classifier: %s' % str(clf_list[0]))
        except: 
            print('t_classify> classifier has not been determined yet.')
        return
    def estimate_performance(res): 
        minLabel, minScore = res['min']
        maxLabel, maxScore = res['max']
        
        print('result> min(label: %s, score: %f)' % (minLabel, minScore))
        print('        max(label: %s, score: %f)' % (maxLabel, maxScore))
        # todo: consider labels
        return (minScore, maxScore)
    def describe_classifier(clf):
        myName = '?'
        try: 
            myName = clf.__class__.__name__ 
        except: 
            print('warning> Could not access __class__.__name__ ...')
            myName = str(clf)
        print('... classifier name: %s' % myName)
        return

    import evaluate, seqparams
    import classifier.utils as cutils
    from tset import TSet, loadTSetCombined

    from sklearn.preprocessing import LabelEncoder
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, SGDClassifier
    
    from sklearn.preprocessing import StandardScaler, MaxAbsScaler # feature scaling for sparse repr. 

    # [params]
    #    set is_augmented to True to use augmented training set
    # tsHandler.config(cohort='CKD', seq_ptype='regular', is_augmented=True) # 'regular' # 'diag', 'regular'
    mode = kargs.get('mode', 'multiclass')  # values: 'binary', 'multiclass'
    param_grid = None
    if mode.startswith('bin'): # binary class (need to specifiy positive classes to focus on)
        return t_binary_classify(**kargs)    

    ### training document vectors 
    # t_model(corhot='CKD', seq_ptype=seq_ptype, load_model=True)

    ### classification 
    clf_list = []

    ### choose classifier 

    # 1. logistic 
    # clf = LogisticRegression(class_weight='balanced', solver='saga', penalty='l2')

    # 2. random forest 
    # clf = RandomForestClassifier(n_jobs=5, random_state=53)  # n_estimators=500, 

    # 3. SGD classifier
    #    when tol is not None, max_iter:1000 by default
    # clf = SGDClassifier(loss='log', penalty='elasticnet', max_iter=1500, tol=1e-3) # l1_ratio: elastic net mixing param: 0.15 by default
    clf = choose_classifier(name=kargs.pop('clf_name', 'rf')) # rf: random forest
    clf_list.append(clf)  # supports multiclass 

    # gradient boosting: min_samples_split=2
    # param_grid = {'n_estimators':[500, 100], 'min_samples_split':[2, 5], 'max_features': [None, 2, 5], 'max_leaf_nodes': [None, 4]}
    # clf_list.append( GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, random_state=53, max_features=2, max_leaf_nodes=4) )
    # clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, random_state=53, max_features=2, max_leaf_nodes=4)

    # random_state = np.random.RandomState(0)
    # clf = svm.SVC(kernel='linear', probability=True, random_state=random_state)

    focusedLabels = None # None to include ALL  # focus(), merge()
    classesOnROC = ['CKD Stage 1', 'CKD Stage 3', 'CKD Stage 5', ] # assuming that CKD Stage 3a and 3b are merged into CKD Stage 3

    # can provide customized training set (ts); if None, then ts is automatically determined by tsHanlder.load()
    # ts(cohort, d2v, ctype, meta('A', 'D', ), index)
    ts_dtype = kargs.get('tset_dtype', 'dense')
    maxNPerClass = kargs.get('n_per_class', None)
    userFileID = kargs.get('meta', tsHandler.meta) 
    tDropControlGroup = kargs.get('drop_ctrl', False)

    experimental_settings()
    if ts_dtype.startswith('d'): 

        # [note] set n_per_class to None => subsampling now is delegated to makeTSetCombined() 
        # load, (scale), modify, subsample
        ts = loadTSetCombined(label_map=seqparams.System.label_map, 
               n_per_class=maxNPerClass, 
               drop_ctrl=tDropControlGroup) # all parameters should have been configured

        X, y = TSet.toXY(ts)
        summary(ts=ts)
        ts = None; gc.collect()

        n_samples = kargs.get('n_samples', X.shape[0])
        n_classes = kargs.get('n_classes', len(np.unique(y)))
        print('t_classify> X: %s | n_classes=%d' % (str(X.shape), n_classes))

        # subsampling 
        # if maxNPerClass: 
        #     ts = subsample2(ts, n=maxNPerClass)

        # precedence: classifier_name -> classifier
        
        result_set = []
        for clf in clf_list: 
            describe_classifier(clf)
            ## a. train-dev-test splits 
            # sampling.splitDataPerClass(y, ratios=[0.8, 0.1, 0.1])

            ## b. CV 
            res = multiClassEvaluate(X=X, y=y, classifier=clf, 
                    focused_labels=None, 
                    roc_per_class=classesOnROC,
                    param_grid=None, 
                    label_map=seqparams.System.label_map, # use sysConfig to specify
                    meta=userFileID, identifier=None) # use meta or identifier to distinguish different classificaiton tasks; set to None to use default
            result_set.append(res)
    else: # sparse
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

        # [test]
        # clf_list = [LogisticRegression(class_weight='balanced', solver='saga', penalty='l1'), ]
        summary(X=X, y=y)
        result_set = []
        for clf in clf_list: 
            res = multiClassEvaluate(X, y, classifier=clf, 
                    focused_labels=None, 
                    roc_per_class=classesOnROC,
                    param_grid=None, 
                    label_map=seqparams.System.label_map, # use sysConfig to specify
                    meta=userFileID, identifier=None) # use meta or identifier to distinguish different classificaiton tasks; set to None to use default
        result_set.append(res)

    minscore, maxscore = estimate_performance(result_set[0])
    # [todo] loop through cartesian product of seq_ptype and d2v_method? 

    return (minscore, maxscore)

# [helper]
def show_weights(model):
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights) 
    return 

# [helper]
def reset_weights_dense(model): 
    """
    Reinitialize weights in the model. 

    Note
    ----
    Only tested on fully-connected layers (i.e. Dense); may not work for other types of layers. 
    """
    from keras import backend as K
    from keras.layers import Dense

    session = K.get_session()
    for layer in model.layers:
        if isinstance(layer, Dense):
            old = layer.get_weights()
            layer.W.initializer.run(session=session)
            layer.b.initializer.run(session=session)
            print(np.array_equal(old, layer.get_weights())," after initializer run")
        else:
            print(layer, "not reinitialized")

    return model
# [helper]
def reinitialize_weights(model, from_file=None):
    """
    Reinitialize weights in the model.

    Note 
    ----

    """
    from keras.initializers import glorot_uniform  # an initializer of your choice

    if from_file is None: 
        # [todo]
        # [err] Cannot evaluate tensor using eval(): No default session is registered. Use 'with sess.as_default()' 
        #       or pass an explicit session to eval(session=sess) 
        initial_weights = model.get_weights()
        new_weights = [glorot_uniform()(w.shape).eval() for w in initial_weights]
        model.set_weights(new_weights)
    else: 
        assert os.path.exists(from_file), "Invalid init_weights: %s" % from_file
        model.load_weights(from_file)

    return model

def t_deep_classify(**kargs):
    """

    Memo
    ----
    1. Example training sets: 
        a. trained with labeled data only (cohort=CKD, meta=smallCKD)
           <prefix>/tpheno/seqmaker/data/CKD/combined/tset-n0-IDregular-pv-dm2-smallCKD-GCKD.csv

    2. training set params: 
        ts = TSet.load(cohort=tsHandler.cohort, 
                       d2v_method=tsHandler.d2v, 
                       seq_ptype=tsHandler.ctype, 
                       suffix=tsHandler.meta, 
                       index=index,  # CV index (if no CV, then 0 by default)
                       dir_type=tsHandler.dir_type) # [params] index

    *3. Import dnn_utils for example networks. 
    """
    def validate_classes(no_throw=True): 
        n_classes = np.unique(y_train).shape[0]
        n_classes_test = np.unique(y_test).shape[0]
        print('t_classify> n_classes: %d =?= n_classes_test: %d' % (n_classes, n_classes_test))
        return n_classes
    def transform_label(l, positive=None):
        if positive is None: positive = positive_classes # go to the outer environment to find its def
        y_ = labeling.binarize(l, positive=positive) # [params] negative
        return y_ 
    def relabel(): 
        # CKD stage classes
        lmap = {}
        lmap['Control'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
        lmap['CKD Stage 5'] = ['ESRD after transplant', 'ESRD on dialysis', ]
        lmap['CKD Stage 3'] = ['CKD Stage 3a', 'CKD Stage 3b', ]
        print('  + Relabeling the data set according to the following map:\n%s\n' % lmap)
        return lmap
    def subsample(X, y, n=None, sort_index=False, random_state=53):
        """
        Subsampling operation applied after doc vec X is derived. 
        """
        if n is not None: 
            idx = cutils.samplePerClass(y, n_per_class=n, sort_index=sort_index, random_state=random_state)
            print('  + X (dim0: %s -> dim: %s)' % (X.shape[0], len(idx)))
            return (X[idx], y[idx])
        return (X, y)
    def check_tset(ts): 
        target_field = TSet.target_field
        n_classes = 1 
        no_throw = kargs.get('no_throw', True)
        if ts is not None and not ts.empty: 
            # target_field in ts.columns # o.w. assuming that only 1 class 
            n_classes = len(ts[target_field].unique())  
            print('t_classify> number of classes: %d' % n_classes)
        else:
            msg = 't_classify> Warning: No data found (cohort=%s)' % kargs.get('cohort', 'CKD')
            if no_throw: 
                print msg 
            else: 
                raise ValueError, msg
    def summary(ts=None, X=None, y=None): 
        # experimental_settings
        msg = "... cohort=%s, ctype=%s, d2v=%s, meta=%s\n" % \
                (tsHandler.cohort, tsHandler.ctype, tsHandler.d2v_method, tsHandler.meta)
        msg += "  + is_simplified? %s, ... \n" % tsHandler.is_simplified
        msg += '  + classification mode: %s\n' % mode
        msg += '  + classifiers:\n%s\n' % clf_list
        msg += '  + training set type:%s\n' % ts_dtype

        nrow = n_classes = -1
        if ts is not None: 
            nrow = ts.shape[0]
            n_classes = len(ts[TSet.target_field].unique())
        else: 
            assert X is not None
            nrow = X.shape[0]
            n_classes = len(np.unique(y)) if y is not None else 1
        msg += '  + training set dim:%d\n' % nrow 
        msg += '  + n classes:%d\n' % n_classes

        print msg 
        return 
    def experimental_settings(): 
        # import vector
        print('\n<<< Experimental Settings >>>')
        print('\n   + tset type %s, maxNPerClass: %s, drop control? %s' % \
            (kargs.get('tset_dtype', 'dense'), kargs.get('n_per_class', None), kargs.get('drop_ctrl', False)))

        print('  + cohort: %s, ctype: %s\n' % (tsHandler.cohort, tsHandler.ctype))
        print('  + D2V: %s, params> window: %s, n_features: %s' % (tsHandler.d2v, vector.D2V.window, vector.D2V.n_features))
        print('       + n_iter: %d, min_count: %d\n' % (vector.D2V.n_iter, vector.D2V.min_count))
        print('  + userFileID: %s' % kargs.get('meta', tsHandler.meta))

        try: 
            print('\n... data: ')
            print('  + n_timesteps: %d, n_features: %d' % ())
            print('  + reshaped X: %s | n_classes=%d' % (str(X.shape), n_classes))
        except: 
            pass
        try: 
            print('\n... params (model selection): ')
            print('  + epochs: %d, batch_size: %d' % NNS.epochs_ms, NNS.batch_size_ms)
            print('\n... params (after model selection)')
            print('  + epochs: %d, batch_size: %d' % NNS.epochs, NNS.batch_size)
        except: 
            pass
        return
    def estimate_performance(res): 
        from seqUtils import format_list
        missing = []
        # res: a dictionary with keys: {min, max, micro, macro, loss, accuracy, auc}
 
        minLabel, minScore = res['min']  # min auc score among all classes
        maxLabel, maxScore = res['max']  # max auc score among all classes

        if res.has_key('min_err'): 
            print('result> min(label: %s, score: %f), err: %s' % (minLabel, minScore, res['min_err']))
        if res.has_key('max_err'): 
            print('        max(label: %s, score: %f), err: %s' % (maxLabel, maxScore, res['max_err']))

        print('result> other performance metrics ...')
        missing = []
        for metric in ('micro', 'macro', 'loss', 'acc', 'auc_roc', ): 
            if res.has_key(metric): 
                 print('    + metric=%s => %f (err: %s)' % (metric, res[metric], res['%s_err' % metric]))
            else: 
                missing.append(metric)
        
        if missing: print('... missing metrics: %s' % format_list(missing))
        # todo: consider labels
        return (minScore, maxScore)
    def load_tset(cv_id=0, meta=None):  # cohort, seq_ptype, d2v_method (focused_labels, label_map)
        ### tsHandler has to be configured first, use tsHandler.config()
        # this should conform to the protocal in makeTSetCombined()

        if kargs.has_key('X') and kargs.has_key('y'): 
            X, y = kargs['X'], kargs['y']
        else: 
            ts = kargs.get('ts', None)
            if ts is None: 
                print('  + automatically loading training set (cohort=%s, d2v=%s, ctype=%s)' % \
                         (tsHandler.cohort, tsHandler.d2v, tsHandler.ctype))
                # config()
                ts = tsHandler.load(cv_id, meta=meta)  # opt: focus_classes, label_map
                ts = modify_tset(ts)  # <- focused_labels, label_map
                assert ts is not None and not ts.empty, "multiClassEvaluate> Null training set."
                profile(ts)
            # allow loadTSet-like operations to take care of this
            X, y = TSet.toXY(ts)
        tsHandler.profile2(X, y)
        return (X, y)
    def profile(ts): 
        return tsHandler.profile(ts)   
    def get_tset_dir(): 
        return tsHandler.get_tset_dir()
    def modify_tset(ts): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Others"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            ts = focus(ts, labels=focused_classes, other_label='Control')
        lmap = kargs.get('label_map', None)
        if lmap: # relabel so that classes only retain the main CKD stages
            print('  + before re-labeling ...')
            profile(ts)
            ts = merge(ts, lmap)
            print('  + after re-labeling ...')
            profile(ts)
        return ts 
    def describe_classifier(clf):
        myName = '?'
        try: 
            myName = clf.__class__.__name__ 
        except: 
            print('warning> Could not access __class__.__name__ ...')
            myName = str(clf)
        print('... classifier name: %s' % myName)
        return
    def create_model(n_units=100): # closure: n_timeteps, n_features, n_classes
        # def
        model = Sequential()
        model.add(LSTM(n_units, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(0.2))
        model.add(Dense(n_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', ])
        return model
    def baseline_model(n_units=100): # closure: n_timeteps, n_features, n_classes
        # create model
        model = Sequential()
        model.add(Dense(n_units, input_dim=n_features, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))
        
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    def reshape3D(X): 
        # n_samples = kargs.get('n_samples', X.shape[0])
        # n_timesteps = kargs.get('last_n_visits', 10)
        # n_features = kargs.get('n_features', vector.D2V.n_features * 2 if tsHandler.d2v in ('pv-dm2', ) else vector.D2V.n_features)
        print('  + reshape(X) for LSTM: n_samples: %d, nt: %d, nf: %d' % (n_samples, n_timesteps, n_features))
        return X.reshape((n_samples, n_timesteps, n_features))
    def save(model):
        identifier = kargs.get('identifier', None) 
        if identifier is None: identifier = make_file_id()
        NNS.save(model, identifier=identifier) # params: model_name
        return
    def load(): 
        identifier = kargs.get('identifier', None) 
        if identifier is None: identifier = make_file_id()
        return NNS.load(identifier=identifier) 
    def rank_performance(scores, gaps, param, metrics_ordered=['loss', 'acc', 'auc_roc', ]): # <- score_map, grid_scores
        
        # score_map = {0:'loss', 1:'accuracy', 2:'auc_roc'}  
        # [update] grid_scores
        for si, metric in enumerate(metrics_ordered):
            setting = {'n_units': param['n_units'], 'dropout_rate': param['dropout_rate'], }
            pmetric = (setting, scores[si], gaps[si])  # [params] modify the desire performance measures here
            grid_scores[metric].append(pmetric)

        return grid_scores
    def rank_model(target_metric, metrics_ordered=['loss', 'acc', 'auc_roc', ], score_pos=1):  # given grid_scores, rank them 
        # [todo]
        if metrics_ordered is None: metrics_ordered = NNS.metrics_ordered

        rank_min = [0, ]  # the indices for which the higher the rank, the smaller the score
        rank_max = [1, 2]   # model.metrics_names[1]
        print('result> performance scores ...\n')

        opt = opt_score = opt_gap = None
        
        # [design]
        max_loss = 0.7
        min_acc, min_auc = 0.8, 0.8

        topN = 5 # keep track of how many times a particular setting was chosen among top N
        popularModels = collections.Counter()
        for si, metric in enumerate(metrics_ordered):
            
            # A. sort by performance score
            # grid_scores[si] = sorted(grid_scores[si], key=lambda x:x[score_pos], reverse=False if metric.startswith('loss') else True)
            # print('verify> full ranking:\n%s\n' % grid_scores[si])

            # B. sort in terms of gaps
            candidates = sorted(grid_scores[metric], key=lambda x:x[score_pos+1], reverse=False) # always the smaller the better
            n_models = len(candidates)
            # print('verify> full ranking (n_models=%d):\n%s\n' % (len(candidates), candidates))
            
            # policy: rank according to gap (between training perfomrance and validation performance, the smaller the better)
            #    subject to: acc >= 0.9, auc >= 0.9 
            candidates2 = []
            for candidate in candidates: 
                score = candidate[score_pos]
                if metric.startswith('loss'): 
                    if score <= max_loss: candidates2.append(candidate)
                elif metric == 'acc': 
                    if score >= min_acc: candidates2.append(candidate)
                elif metric == 'auc_roc':
                    if score >= min_auc: candidates2.append(candidate)
            grid_scores[metric] = candidates2
            
            n_models_final = len(grid_scores[metric])
            print('... performance ranking (n_models:%d -> %d):\n%s\n' % (n_models, len(candidates2), candidates2))
            assert n_models_final > 0

            # most popular (e.g. top 5) across different metrics? 
            configs = []
            for pmetric in grid_scores[metric][:topN]:  # pmetric: (setting, scores[si], gaps[si])
                setting = pmetric[0]
                assert len(setting) >= 2
                e = tuple([(k, v) for k, v in setting.items()]) 
                configs.append(e)
            popularModels.update(configs)     

            best_scores = grid_scores[metric][0]
            print('... under metric (%s), best score: %f, gap: %f' % (metric, best_scores[score_pos], best_scores[score_pos+1]))
            print('... model config: %s\n' % best_scores[0])
            
            if target_metric == si or target_metric.startswith(metric):
                opt = best_scores[0]  # a dictionary
                
        topNFinal = 10
        print('result> popular %d model (out of %d metric-neutral options with topN=%d) ...' % (topNFinal, len(popularModels), topN))
        for setting, n_selected in popularModels.most_common(topNFinal):
            print('  + (n_selected=%d) model: %s' % (n_selected, setting))

        print('result> best configuration:\n%s\n' % opt)   # {'n_units': 200, 'dropout_rate': 0.5}
        return opt

    import evaluate, seqparams, vector
    from seqConfig import tsHandler
    import classifier.utils as cutils
    from tset import TSet, loadTSetCombined
    from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, SGDClassifier
    from sklearn.preprocessing import StandardScaler, MaxAbsScaler # feature scaling for sparse repr. 

    # deep learning packages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.wrappers.scikit_learn import KerasClassifier  ###
    from keras.utils import np_utils
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, GridSearchCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import ParameterGrid
    from functools import partial

    # from dnn_utils import

    # [params]
    #    set is_augmented to True to use augmented training set
    # tsHandler.config(cohort='CKD', seq_ptype='regular', is_augmented=True) # 'regular' # 'diag', 'regular'
    mode = kargs.get('mode', 'multiclass')  # values: 'binary', 'multiclass'
    param_grid = None 

    ### training document vectors 

    focusedLabels = None # None to include ALL  # focus(), merge()
    classesOnROC = ['CKD Stage 1', 'CKD Stage 3', 'CKD Stage 5', ] # assuming that CKD Stage 3a and 3b are merged into CKD Stage 3

    # can provide customized training set (ts); if None, then ts is automatically determined by tsHanlder.load()
    # ts(cohort, d2v, ctype, meta('A', 'D', ), index)
    ts_dtype = kargs.get('tset_dtype', 'dense')
    maxNPerClass = kargs.get('n_per_class', None)
    userFileID = kargs.get('meta', tsHandler.meta) 
    tDropControlGroup = kargs.get('drop_ctrl', False)

    n_timesteps = kargs.get('last_n_visits', 10)
    n_features = kargs.get('n_features', vector.D2V.n_features * 2 if tsHandler.d2v in ('pv-dm2', ) else vector.D2V.n_features)

    ### Add classification models
    clf_list = []
    # define + compile
    # clf = choose_classifier(name=kargs.pop('clf_name', 'lstm'), 
    #           epochs=kargs.get('epochs', 200), batch_size=kargs.get('batch_size', 16)) 
    # clf_list.append(clf)  # supports multiclass 

    # experimental_settings()
    result_set = []
    minscore = maxscore = -1
    grid_scores = {}
    tModelSelection = True
    if ts_dtype.startswith('d'): 

        # [note] set n_per_class to None => subsampling now is delegated to makeTSetCombined() 
        # load, (scale), modify, subsample

        # training set file parameters are determined via seqConfig.tsHandler
        ts = loadTSetCombined(label_map=seqparams.System.label_map, 
               n_per_class=maxNPerClass, 
               drop_ctrl=tDropControlGroup) # all parameters should have been configured
        summary(ts=ts)
        print('d_classify> dim(ts): %s > n_timesteps: %d, n_features: %d' % (str(ts.shape), n_timesteps, n_features))
       
        X, y = TSet.toXY(ts)
        ts = None; gc.collect()

        n_samples = kargs.get('n_samples', X.shape[0])
        n_classes = kargs.get('n_classes', len(np.unique(y)))
        print('  + dim(X <- ts): %s' % str(X.shape))

        # to 3D in order to use LSTM
        #   [n_samples, n_timesteps, n_features]
        X = X.reshape((n_samples, n_timesteps, n_features))
        print('d_classify> reshaped X: %s | n_classes=%d' % (str(X.shape), n_classes))
        
        # subsampling 
        # if maxNPerClass: 
        #     ts = subsample2(ts, n=maxNPerClass)

        # define + compile
        # model = KerasClassifier(build_fn=gen_lstm(n_units=100, n_timesteps=n_timesteps, n_features=n_features, n_classes=n_classes), 
        #            epochs=200, batch_size=16, verbose=1)

        # A. Define a fixed model: define + compile
        # model = choose_classifier(n_timesteps=n_timesteps, n_features=n_features, n_classes=n_classes, 
        #     n_units=50, dropout_rate=0.2,  # params for model definition
        #     epochs=200, batch_size=32)  # params for model.fit()
        # clf_list.append(model)
        targetMetric = kargs.get('target_metric', 'loss') # model selection and policy for saving the model
        if not tModelSelection: 
            # model = load()
            n_units, dropout_rate = 200, 0.5
            model = make_lstm(n_units=500, n_timesteps=n_timesteps, n_features=n_features, 
                n_classes=n_classes, dropout_rate=0.2) # compiled model
            clf_list.append(model)
            # fit (+evaluate)
        else: 
            # if X_val is None:
            #     print('d_classify> Warning: No separate validation set provided!')
            #     X_val, y_val = X, y

            # B. Model selection
            param_grid = {'n_units': [50, 100, 200, 300, 400, 500], 'dropout_rate': [0.2, 0.3, 0.5, 0.6, ]}
            NNS.epochs_ms = kargs.get('epochs_ms', 100)  # previously: 100
            NNS.batch_size_ms = kargs.get('batch_size_ms', 32)
            NNS.patience_ms = kargs.get('patience_ms', 20)
        
            test_ratio = 0.3
            score_map = {0:'loss', 1:'accuracy', 2:'auc'}
            score_index = 1  # score_index: 0/loss, 1/accuracy, 2/auc
        
            # Run model selection
            grid_scores = {metric:[] for metric in NNS.metrics_ordered}  # ['loss', 'acc', 'auc_roc', ]
            experimental_settings()
            for param in list(ParameterGrid(param_grid)): 
                print('\nmodel_selection> trying %s ...\n' % param) 

                # use KerasClassifier Wrapper
                # model = makeClassifier(X, y, build_fn=build_fn_lstm, 
                #             n_layers=n_layers, n_units=n_units, dropout_rate=dropout_rate, # params for architecture 
                #             callbacks=None)  # callbacks: None to use default 

                model = make_lstm(n_units=param['n_units'], n_timesteps=n_timesteps, n_features=n_features, 
                            n_classes=n_classes, dropout_rate=param['dropout_rate'], save_init_weights=False)

                # params: random_state
                # scores and gaps are in the following order: ['loss','acc', 'auc_roc']
                scores, gaps = modelEvaluate0(X, y, model=model, focused_labels=None, roc_per_class=classesOnROC, 
                                    meta=userFileID, identifier=None, 
                                    test_size=test_ratio,  # only applies when using train_test_split
                                    validation_split=test_ratio, 
                                    patience=NNS.patience_ms, 
                                    epochs=NNS.epochs_ms, batch_size=NNS.batch_size_ms)  # score_index: 0/loss, 1/accuracy, 2/auc
            
                # grid_scores[i]: (setting, scores[si], gaps[si])
                rank_performance(scores, gaps, param=param)  # modify: grid_scores; other params: metrics_ordered
                
                # score = scores[score_index]
                # grid_score.append((param['n_units'], param['dropout_rate'], score))

            # choose metric given by 'score_index' as the final measure
            opt = rank_model(target_metric=targetMetric)  # rank hyperparams and their scores

            print('model_select> opt model:\n%s\n' % opt)
            n_units, r_dropout = opt['n_units'], opt['dropout_rate']
 
            # choose_classifier(build_fn=partial(make_lstm, ...))
            model = make_lstm(n_units=n_units, n_timesteps=n_timesteps, n_features=n_features, 
                n_classes=n_classes, dropout_rate=r_dropout) # compiled model
            clf_list.append(model)

        ### model defined and weights trained 

        # other params: outputfile, outputdir
        # <note> if 'identifier' is not None, then its value will serve as the file ID (i.e. meta will be ignored)
        
        # [params] post-model-selection parameters
        #          precedence: classifier_name -> classifier
        NNS.patience = kargs.get('patience', 20)  # this controls EarlyStopping    
        NNS.epochs = kargs.get('epochs', 200) 
        NNS.batch_size = kargs.get('batch_size', 32)
        
        nTrials = 10
        for clf in clf_list: 
            describe_classifier(clf)

            # simple evaluation
            # scores = model.evaluate(X_test, y_test, verbose=0)

            ### a. train test split
            # other params: outputfile, outputdir
            # res = modelEvaluate(X, y, model=model, focused_labels=None, roc_per_class=classesOnROC, 
            #         meta=userFileID, identifier=None, 
            #         patience=NNS.patience, epochs=NNS.epochs, batch_size=NNS.batch_size, 
            #         save_model=True)  # pass random_state to fix it

            ### a2. multiple trails 
            # [todo] reset weights
            res = modelEvaluateBatch(X, y, model, focused_labels=None, roc_per_class=classesOnROC, 
                    meta=userFileID, identifier=None, 
                    patience=NNS.patience, epochs=NNS.epochs, batch_size=NNS.batch_size, 
                    save_model=True, n_trials=nTrials, target_metric=targetMetric, init_weights=NNS.init_weights)   

            # ## b. CV
            # res = multiClassEvaluate(X=X, y=y, 
            #         classifier=clf, 
            #         focused_labels=None, 
            #         roc_per_class=classesOnROC,
            #         param_grid=None, 
                  
            #         evaluation='split', ratios=[0.7, ]  # non-CV evaluation (e.g. train test split)

            #         label_map=seqparams.System.label_map, # use sysConfig to specify
            #         meta=userFileID, identifier=None, # use meta or identifier to distinguish different classificaiton tasks; set to None to use default
            #         ) 

            result_set.append(res)
            
        minscore, maxscore = estimate_performance(result_set[0])
        # if len(grid_scores) > 0: print('result> best combo (n_unit, rate => score): %s' % str(grid_scores[0]))
    else: 
        raise ValueError, "LSTM mode does not support sparse training set."

    
    # [todo] loop through cartesian product of seq_ptype and d2v_method? 

    return (minscore, maxscore)

def sysConfig(cohort, d2v_method='pv-dm2', ctype='regular', **kargs):
    """
    Configure system-wide paramters applicable to all modules that import seqparams.

    Params
    ------
    """
    def relabel():  # ad-hoc
        # CKD stage classes
        lmap = {}
        lmap['Others'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
        lmap['CKD Stage 5'] = ['ESRD after transplant', 'ESRD on dialysis', ]
        lmap['CKD Stage 3'] = ['CKD Stage 3a', 'CKD Stage 3b', ]
        return lmap

    import seqparams
    from seqparams import System as sysparams
    from seqConfig import tsHandler 
    import vector

    sysparams.cohort = cohort # system parameters are shared across modules (e.g. vector module assess routine depends on cohort value)
    sysparams.label_map = relabel()

    # configure d2v params
    wsize = seqparams.D2V.window = 10
    # assert vector.D2V.window == wsize
    n_features = seqparams.D2V.n_features = 50   # 20*
    n_iter = seqparams.D2V.n_iter = 20
    # assert vector.D2V.n_features == n_features

    # training set parameters 
    if d2v_method is None: d2v_method = vector.D2V.d2v_method
    user_file_descriptor = kargs.get('meta', 'test')

    tsHandler.config(cohort=cohort, d2v_method=d2v_method, 
                    seq_ptype=kargs.get('seq_ptype', ctype),
                    **kargs)  # is_augmented/False, is_simplified/False, dir_type/'combined'
    return tsHandler

def process_tset(**kargs):
    """
    Test cases for temporal phenotype project. 
    """ 
    def display(docs, nmax=10):
        tD = random.sample(docs, min(len(docs), 10))
        for i, doc in enumerate(docs):  
            if i < nmax: 
                print('[%d, size=%d] %s' % (i, len(doc), doc))
        print('-' * 80)
        return

    # cohort_name = 'CKD'
    ret = {}  # [output]

    # model parameters
    last_n_visits = kargs.pop('last_n_visits', 20)   # after d2v is derived, represnet each document by the vectors of the last n visits
    maxNTokensPerVisit = kargs.pop('max_visit_length', 100)

    tsHandler = sysConfig(cohort='CKD', d2v_method='pv-dm2', ctype='regular', **kargs)  # meta set to smallCKD  ... 1a
    print('check> sysConfig complete ... meta? %s' % tsHandler.meta)
    
    ### load input document 
    tSegmentVisit = kargs.get('segment_by_visit', True) # kargs.pop('segv', True) 
    docMode = 'visit'  # only applies when segment_by_visit is True
    
    prefix = os.getcwd()
    inputdir = os.path.join(prefix, 'data/%s' % tsHandler.cohort)
    cohort_corpus_source = 'condition_drug_labeled_seq-CKD.csv'

    ret = load_data(cohort=tsHandler.cohort, inputdir=inputdir, inputfile=cohort_corpus_source, 
        segment_by_visit=tSegmentVisit, max_visit_length=maxNTokensPerVisit, 
        mode=docMode, 
        label_loc=True, 
        to_str=False)
    D, L, T, visitDocIDs, docIDs = ret['D'], ret['L'], ret['T'], ret['visitDocIDs'], ret['docIDs']
    print('> D:\n%s\n' % random.sample(D, min(len(D), 5)))
    # display(D)

    # tD = docToInteger(tD, policy='exact'); display(tD)
    # tD = pad_sequences(sequences=D, maxlen=None, value=0); display(tD)

    # train small model 
    if tSegmentVisit: 
        X, y = makeTSetVisit(D, T, L, visitDocIDs, last_n_visits=last_n_visits, 
            cohort=tsHandler.cohort, meta=tsHandler.meta, 
            test_model=False, load_model=True)  # meta still 'D' ... 1b (value was not received from 1a)
    else: 
        X, y = makeTSetDoc(D, T, L, docIDs, cohort=tsHandler.cohort, meta=tsHandler.meta, 
            test_model=False, load_model=True)
    
    ret['data'] = (X, y)
    ret['last_n_visits'] = last_n_visits

    return ret

def load_data_mnist():     
    """

    Memo
    ----
    1. deep_learn/ch19
    """
    from keras import backend as K

    K.set_image_dim_ordering('th')
    
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
    
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    return (X_train, y_train, X_test, y_test)

# Larger CNN for the MNIST Dataset
def t_larger_cnn(num_classes):
    # create model
    model = Sequential()

    # 30 feature detectors, 5-by-5 filter | input: 1 channel 28 by 28
    model.add(Convolution2D(30, 5, 5, input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(15, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build(): 
    
    X_train, y_train, X_test, y_test = load_data_mnist() # load data
    n_classes = y_test.shape[1]  # this only makes sense because np_utils.to_categorical(y_test)

    # build the model
    model = t_larger_cnn(num_classes=n_classes)

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    
    print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

    return

def test(**kargs): 
    """
    Test cases for generic deep neural nets. 
    """

    # AUC metric
    # demo_auc_roc()

    # CNN example #1 
    # build()

    ### Misc 
    scores = {'min': [('Stage 3', 0.76), ('Stage 5', 0.99), ('Stage 5', 0.90)], 
              'acc': [0.77, 0.88, 0.99, 0.92, 0.88, 0.86, ]}
    res = analyze_performance(scores, n_resampled=100)
    for k, v in res.items(): 
        print('... %s => %s' % (k, v))
    return

def test2(**kargs): 
    
    # tokenization 
    # t_tokenize()
    tNNet = False
    tSegmentVisit = True if tNNet else False

    userFileID = 'smallCKD-Visit'
    if not tSegmentVisit: 
        userFileID = 'smallCKD-Doc'

    # process documents and training set
    tset = process_tset(last_n_visits=50, max_visit_length=100, 
        segment_by_visit=tSegmentVisit, meta=userFileID)
    # X, y = tset['data']

    ### machine learning methods 
    if tNNet: 
        batch_size_ms = 1
        batch_size = 1

        print("info> each doc is repr by the last %d visits" % tset['last_n_visits'])
        m, M = t_deep_classify(clf_name='lstm', epochs=80, batch_size=batch_size, 
            epochs_ms=60, batch_size_ms=batch_size_ms,   # other params: patience_ms,
            patience_ms=20, patience=25, 
            metric_ms='loss',    # use this metric for model selection (minimizing loss, maximizing acc or auc_roc) 
            last_n_visits=tset['last_n_visits'])

        # also see 
        # t_larger_cnn(num_classes)
    else: 
        m, M = t_classify(mode='multiclass', n_per_class=None, tset_dtype='dense', drop_ctrl=False, 
                          clf_name='gradientboost')

    return 

if __name__ == "__main__": 
    # test()

    test2()

    

