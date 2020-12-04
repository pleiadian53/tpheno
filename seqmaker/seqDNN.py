# encoding: utf-8

# non-interactive mode
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt
plt.style.use('ggplot')

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# support roc_auc in model evaluation
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping

from sklearn import preprocessing

# from system.utils import div

from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping
from keras.models import model_from_json

import time, os, sys, math, re, gc
import numpy as np

### project module 
import seqparams 
import vector
import dnn_utils
from tsHandler import tsHandler
from sampler import sampling


class NNS(dnn_utils.NNS): 
    model_name = 'lstm'

class CNNSpec(object): 
    pass 

class LSTMSpec(object): 
    n_layers = 1 
    n_timesteps = 10 
    
    n_features = 200 
    n_units = n_features 

    r_dropout = 0.2 
    optimizer = 'adam'

    patience = patience_ms = 100 
    batch_size = batch_size_ms = 16
    epochs = epochs_ms = 1000 
    shuffle = True

    r_validation_split = 0.3

    metric = 'auc_roc'

    # for multiple instances of LSTM models? 
    def __init__(self, n_timesteps=None, n_features=None, n_units=None):
        self.n_timesteps = LSTMSpec.n_timesteps if n_timesteps is None else n_timesteps
        self.n_features = LSTMSpec.n_features if n_features is None else n_features

        self.n_units = self.n_features if n_units is None else n_units 
        # self.n_layers = n_layers

    @staticmethod
    def params_summary(): 
        n_epochs_no_improv_tolerance = LSTMSpec.patience
        validation_split_ratio = LSTMSpec.r_validation_split

        print('=' * 100)
        print('... early stopping>')
        print("       + patience: %d, model='max'" % n_epochs_no_improv_tolerance)
        print('... model training>')
        print("       + batch_size: %s, epochs: %s, shuffle? %s, metric: %s" % \
            (LSTMSpec.batch_size, LSTMSpec.epochs, LSTMSpec.shuffle, LSTMSpec.metric))
        print("       + validation_split: %f" % LSTMSpec.r_validation_split)
        print('=' * 100)
        return 

def modelEvaluateClassic(X, y, model, **kargs): 
    import modelSelect as ms
    return ms.modelEvaluate(X, y, model, **kargs)
def modelEvaluate(X, y, model, **kargs): 
    """

    Note
    ----
    1. model is to be defined outside this routine, but this results in the need of re-initializing weights 
       under the condition of multiple runs. 

    """
    return dnn_utils.modelEvaluate(X, y, model, **kargs)
def modelEvaluateBatch(X, y, model, **kargs):
    return dnn_utils.modelEvaluateBatch(X, y, model, **kargs) 

def lstmEvaluateChunk(fpath, **kargs): 
    """
    Load the input training data by chunks and incrementally train the LSTM model. 

    Input
    -----
    fpath: path to the training data

    """
    def tune_model(): 
        spec = LSTMSpec(n_timesteps=kargs.get('n_timesteps', 10), n_features=kargs.get('n_features', 200))
        spec.r_dropout = kargs.get('dropout_rate', 0.2)
        return spec
    def reshape3D(X, dims): 
        # n_samples = kargs.get('n_samples', X.shape[0])
        # n_timesteps = kargs.get('last_n_visits', 10)
        # n_features = kargs.get('n_features', vector.D2V.n_features * 2 if tsHandler.d2v in ('pv-dm2', ) else vector.D2V.n_features)  
        assert len(dims) == 3  
        print('(reshape3D) reshape(X) for LSTM > given X: %s > desired: (n_samples, n_timesteps, n_features): %s' % (str(X.shape), str(dims)))
        return X.reshape((n_samples, n_timesteps, n_features))
    def set_optimizer():
        if spec.optimizer == 'sgd':  
            return SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
        return spec.optimizer
    def step_decay(epoch):
        # import math
        initial_lrate = 0.1
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
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
        print('(eval_weights) lookup:\n%s\n' % dict(zip(y_uniq, le.transform(y_uniq))) )

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
    def history_evluation(history, metrics_ordered=None, n_metrics=None):

        # observe model fitting quality (underfitting? overfitting? just right?)
        file_id = kargs.get('identifier', '')
        nlastm = 15
        for mt, mv in [('acc', 'val_acc'), ('loss', 'val_loss'), ]: 
            # [test]
            for m in (mt, mv): 
                print('... ... metric (%s): %s (n=%d)' % (m, history.history[m][-nlastm:], len(history.history[m])))
        
            # generalization error 
            # print('... ... metric (gap): %s' % history.history[mt][-nlastm:]-history.history[mv][-nlastm:])
            save_plot(plot_metric(train=mt, test=mv), identifier=file_id, metric=mt)

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
    def init_model(n_classes): 
        model = kargs.get('model', None)
        if model is None: 
            # condition: LSTM params have been configured
            model = dnn.make_lstm(n_layers=spec.n_layers, n_units=spec.n_units, n_timesteps=spec.n_timesteps, n_features=spec.n_features, 
                        n_classes=n_classes, dropout_rate=spec.r_dropout, optimizer=set_optimizer()) # compiled model
        return model
    def binarize_label(y_uniq):  # dense layer has as many nodes as number of classes 
        lb = preprocessing.LabelBinarizer()  # this converts to binary codes
        class_labels = np.unique(y_uniq)
        lb.fit_transform(class_labels)
        # print('binarizer> fit %d labels' % len(class_labels))
        return lb # dict(lookup)

    import dnn_utils as dnn
    import pandas as pd
    from dnn_utils import NNS
    from tset import TSet
    from tsHandler import loadTSetChunk2
    import plotUtils as putils
    from seqparams import Graphic
    import modelSelect as ms
    from keras.callbacks import LearningRateScheduler
    from keras.optimizers import SGD

    # import time
    nTrials = kargs.get('n_trials', 1)
    kargs['target_metric'] = targetMetric = kargs.get('target_metric', 'loss')

    kargs['ref_performance'] = ref_performance = 0.5
    scores, finalRes = {}, {}
    tInitPerTrial = kargs.get('reinit_weights', True)
    tDropControlGroup = kargs.get('drop_ctrl', False)

    # train_subset = kargs.get('train_subset', 10000)
    # tSubset = train_subset < 1.0 or train_subset >= 2

    # customize modelEvaluate
    kargs['conditional_save'] = True  # if True, save only the model that performs better than the previous one
    
    spec = kargs.get('lstm_spec', None)
    if spec is None: 
        spec = tune_model()
    else: 
        assert isinstance(spec, LSTMSpec)

    chunksize = kargs.get('chunksize', 1000)
    # first run: probe the size 
    nAcc = 0
    ylist = []  # estimate the number of classes (needed later to define the NN model)
    Xdim = -1

    # assert os.path.exists(fpath) and os.path.getsize(fpath) > 0, "Invalid training data at:\n%s\n" % fpath

    # [note] use wrapper of pd.read_csv(fpath, chunksize=chunksize, iterator=True) to load chunks of training data
    nrows = ncols = 0
    n_chunks = 0 
    for i, ts in enumerate(loadTSetChunk2(fpath, label_map=seqparams.System.label_map, drop_ctrl=tDropControlGroup)): 
        nAcc += ts.shape[0]

        if i == 0: 
            X, y = TSet.toXY(ts)
            ncols = ts.shape[1]

        y = ts[TSet.target_field].values
        ylist.append(y)
        nrows += ts.shape[0]

        n_chunks += 1

    ts = None; gc.collect()

    # training data statistics
    y = np.hstack(ylist)
    y_uniq = np.unique(y)
    n_classes = len(y_uniq)
    N = nAcc  
    n_features = spec.n_features
    Xdim = (nrows, ncols)
    print('lstmEvaluateChunk> N=%d, n_classes=%d, dim(X)=%s, n_features=%d, n_timesteps:%d=?=%d' % \
        (N, n_classes, str(Xdim), n_features, spec.n_timesteps, ncols/n_features))

    # the smaller the size, the heigher the class weight
    class_weight_dict = eval_weights(y) # this has to come before binarzing
    print('... class weights:\n%s\n' % class_weight_dict)
    
    # control the total number of training data 
    train_ratio = kargs.get('ratio', 0.7)
    train_subset = min(int(N*train_ratio), tsHandler.N_train_max)  # max: 10000
    test_subset = min(N-train_subset, tsHandler.N_test_max)  # max: 5000 
    print('... N_train: %d, N_test: %d' % (train_subset, test_subset))

    # LSTM model setting
    spec.params_summary()
    target_metric = 'val_loss'  # note: monitor <- 'auc_roc' doesn't seem useful
    mode = eval_mode(target_metric)  # higher is better or lower is better? 

    # learing rate schedule
    # lrate = LearningRateScheduler(step_decay)

    ### regularization 
    # early stopping
    # Callback: early stopping, learning rate schedule 
    callback_list = [EarlyStopping(monitor=target_metric, min_delta=0, 
        patience=spec.patience, verbose=1, mode=mode), LearningRateScheduler(step_decay)] # [note] scheduling function (e.g. step_decay())

    # vars: chunksize = kargs.get('chunksize', 1000)
    n_timesteps = spec.n_timesteps # kargs.get('last_n_visits', ncols/n_features)
    scores = {}
    model = None 
    for i in range(nTrials): 

        ### define model 
        model = init_model(n_classes)

        # incrementally train the model
        # [note] use pd.read_csv to read the csv file in chunks of 1000 lines with chunksize=1000 option
        nAcc = 0; history = None
        X_test, y_test = [], []  # hopefully not too big 
        nChunkTrain = nChunkTest = 0

        # use wrapper of pd.read_csv(fpath, chunksize=chunksize, iterator=True) to load chunks of training data
        epochs_div = int(spec.epochs/n_chunks)+5 
        print('(model) n_chunks: %d, epochs: %d => epochs per chunk: %d' % (n_chunks, spec.epochs, epochs_div))
        for i, ts in enumerate(loadTSetChunk2(fpath, label_map=seqparams.System.label_map, drop_ctrl=tDropControlGroup)):

            # preprocess the training set (label remapping)
            X, y = TSet.toXY(ts); ts = None; gc.collect()
            y_uniq = np.unique(y)

            nAcc += X.shape[0] # process this many training instances so far

            n_samples = X.shape[0]
            n_classes_chunk = len(y_uniq)

            X = reshape3D(X, dims=(n_samples, n_timesteps, n_features))
            print('... Chunk #%d > reshaped X: %s | num classes in this chunk: %d' % (i, str(X.shape), n_classes_chunk))
            assert X.shape[0] == len(y)   

            if nAcc <= train_subset:  # this is only approximate
                X_train, y_train = X, y
                lb = binarize_label(y_uniq)
                y_train = lb.transform(y_train)
                history = model.fit(X_train, y_train,
                            validation_split=spec.r_validation_split,  # e.g. 0.3

                            # the higher weight, the higher penalty when misclassified; weights evaluated earlier using the entire class label set
                            class_weight=class_weight_dict, 

                            shuffle=spec.shuffle,
                            batch_size=spec.batch_size, epochs=epochs_div, verbose=1,
                            callbacks=callback_list) 
                nChunkTrain += 1
            else: # testing         
                # ts_test.append(ts)
                X_test.append(X); y_test.append(y)
                nChunkTest += 1
                print('... accumulating test data chunk #%d ...' % nChunkTest)
                
        assert len(X_test) > 0
        assert model is not None, "Null model after training?"
        
        print("lstmEvaluateChunk> assembling a test set from %d chunks' worth of data" % nChunkTest)
        # ts = pd.concat(ts_test, ignore_index=True)
        X_test = np.vstack(X_test)
        y_test = np.hstack(y_test)

        # summarize history for accuracy
        if history is not None: # history: lastest history in the training iteration
            scores_hist, gaps = history_evluation(history, metrics_ordered=NNS.metrics_ordered)  # ['loss','acc', 'auc_roc']

        # y_test = lb.transform(y_test)
        print('lstmEvaluateChunk> Effective training set size: n=%d' % X_test.shape[0])
        # evaluate & save the model
        res = evaluate(X_test, y_test, model=model, **kargs) # params: history

        # update reference preformance: don't save a model unless performance gets better
        ref_performance = res[targetMetric]
        for mt, score in res.items(): 
            if not scores.has_key(mt): scores[mt] = []
            scores[mt].append(score)
    ### end foreach trial
    n_resampled = 100
    res = dnn.analyze_performance(scores, n_resampled=n_resampled)
                
    return res

def lstmEvaluateBatch(X, y, **kargs): 
    """

    Note
    ----
    1. model is defined within the routine and the parameters are specified via an instance of LSTMSpec 

    """
    def tune_model(): 
        spec = LSTMSpec(n_timesteps=kargs.get('n_timesteps', 10), n_features=kargs.get('n_features', 200))
        spec.r_dropout = kargs.get('dropout_rate', 0.2)
        return spec

    # from sampler import sampling
    import dnn_utils as dnn
    # from sklearn.model_selection import train_test_split
    # import time

    nTrials = kargs.get('n_trials', 1)
    kargs['target_metric'] = targetMetric = kargs.get('target_metric', 'loss')
    kargs['ref_performance'] = ref_performance = 0.5
    scores, finalRes = {}, {}

    tInitPerTrial = kargs.get('reinit_weights', True)
    # train_subset = kargs.get('train_subset', 10000)
    # tSubset = train_subset < 1.0 or train_subset >= 2

    # customize modelEvaluate
    kargs['conditional_save'] = True  # if True, save only the model that performs better than the previous one
    
    spec = kargs.get('lstm_spec', None)
    if spec is None: 
        spec = tune_model()
    else: 
        assert isinstance(spec, LSTMSpec)

    for i in range(nTrials): 
        # if tSubset: 
        #     Xs, ys = subset(X, y, train_subset=train_subset)
        #     print('modelEvaluateBatch> trial #%d | N=%d' % (i+1, len(Xs)))

        # **kargs: ratio_train, if specified, only use this fraction of data to train model
        #          train_size, test_size
        res = lstmEvaluate(X, y, spec=spec, **kargs)  # a dictionary with keys: {min, max, micro, macro, loss, acc, auc_roc} 
        
        # update reference preformance: don't save a model unless performance gets better
        ref_performance = res[targetMetric]
        for mt, score in res.items(): 
            if not scores.has_key(mt): scores[mt] = []
            scores[mt].append(score)
        
        # [note] no need to reinit weights here, just redefine the model as if it were a new one
        # if tInitPerTrial: 
        #     model = dnn.reinitialize_weights(model, from_file=NNS.init_weights)

    n_resampled = 100
    res = dnn.analyze_performance(scores, n_resampled=n_resampled)

    return res 

# # learning rate schedule
# def step_decay(epoch):
#     # import math
#     initial_lrate = 0.1
#     drop = 0.5
#     epochs_drop = 10.0
#     lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
#     return lrate

def lstmEvaluate(X, y, spec, **kargs):
    """


    Examples 
    --------
    modelEvaluateBatch2(X, y, model, focused_labels=None, roc_per_class=classesOnROC, 
                        meta=userFileID, identifier=None, 
                        patience=NNS.patience, epochs=NNS.epochs, batch_size=NNS.batch_size, 
                        save_model=True, n_trials=nTrials, 

                        train_subset=train_subset, # only use this fraction of data to train model (for big data)
                        test_ratio=0.3, 

                        target_metric='loss', init_weights=NNS.init_weights)


    """
    def history_evluation(history, metrics_ordered=None, n_metrics=None):

        # observe model fitting quality (underfitting? overfitting? just right?)
        file_id = kargs.get('identifier', '')
        nlastm = 15
        for mt, mv in [('acc', 'val_acc'), ('loss', 'val_loss'), ]: 
            # [test]
            for m in (mt, mv): 
                print('... ... metric (%s): %s (n=%d)' % (m, history.history[m][-nlastm:], len(history.history[m])))
        
            # generalization error 
            # print('... ... metric (gap): %s' % history.history[mt][-nlastm:]-history.history[mv][-nlastm:])
            save_plot(plot_metric(train=mt, test=mv), identifier=file_id, metric=mt)

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
    def step_decay(epoch):
        # import math
        initial_lrate = 0.1
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    def set_optimizer():
        if spec.optimizer == 'sgd':  
            return SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
        return spec.optimizer

    import dnn_utils as dnn 
    from dnn_utils import NNS
    import plotUtils as putils
    from seqparams import Graphic
    import modelSelect as ms
    from keras.callbacks import LearningRateScheduler
    from keras.optimizers import SGD

    # from keras.callbacks import Callback, EarlyStopping

    N = X.shape[0]
    ratio = kargs.get('ratio', 0.7)  # ratio of data used for training
    train_subset = kargs.get('train_subset', int(N*ratio))
    test_subset = kargs.get('test_subset', N-train_subset) 

    # control training size and test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        train_size=train_subset, 
        test_size=test_subset,   # det_test_size(), # accept test_size, ratios
        random_state=kargs.get('random_state', int(time.time())%1000), stratify=y)
    print('lstmEvaluate> dim(X_train):%s, dim(X_test): %s' % (str(X_train.shape), str(X_test.shape))) # (1651, 10, 50), (709, 10, 50)
    print('              dim(y_train):%s, dim(y_test):%s' % (str(y_train.shape), str(y_test.shape)))

    ### define LSTM model 
    y_uniq = np.unique(y)
    n_classes = len(y_uniq)

    # assert isinstance(spec, LSTMSpec)
    model = dnn.make_lstm(n_layers=spec.n_layers, n_units=spec.n_units, n_timesteps=spec.n_timesteps, n_features=spec.n_features, 
                n_classes=n_classes, dropout_rate=spec.r_dropout, optimizer=set_optimizer()) # compiled model

    # the smaller the size, the heigher the class weight
    class_weight_dict = eval_weights(y) # this has to come before binarzing
    print('... class weights:\n%s\n' % class_weight_dict)

    ### regularization 
    # early stopping
    spec.params_summary()
    target_metric = 'val_loss'  # note: monitor <- 'auc_roc' doesn't seem useful
    mode = eval_mode(target_metric)  # higher is better or lower is better? 

    # learing rate schedule
    # lrate = LearningRateScheduler(step_decay)

    # Callback: early stopping, learning rate schedule 
    callback_list = [EarlyStopping(monitor=target_metric, min_delta=0, 
        patience=spec.patience, verbose=1, mode=mode), LearningRateScheduler(step_decay)]

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

    # provide the trained model so that the following routine only need to produce label predictions (y_pred/y_score) from which 
    # ROC plot is computed. 

    # summarize history for accuracy
    scores, gaps = history_evluation(history, metrics_ordered=NNS.metrics_ordered)  # ['loss','acc', 'auc_roc']

    # y_test = lb.transform(y_test)
    res = evaluate(X_test, y_test, model=model, **kargs) # params: history

    return res

def evaluate(X_test, y_test, model=None, **kargs): 
    """
    Given TEST dataset in (X, y)-pair and the trained model, evaluate the model performance
    using a given metric (e.g. accuracy). 

    **kargs
    -------
    model: trained model. Set to None or leave out to load existing model. 

    """
    def get_clf_name(): # can be called before or after model selection (<- select_classifier)
        name = 'nns'
        try: 
            name = model.__name__
        except: 
            print('info> infer classifier name from class name ...')
            # name = str(estimator).split('(')[0]
            name = model.__class__.__name__
        return name
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

            plot_selected_classes=False,  # if False, plot the ROC for all classes instead of just those specified by target_labels
            target_labels=target_labels) 
        return res 
    def general_evaluation(X_test, y_test, trained_model, lb=None, metrics_ordered=None):
        if lb is None: lb = binarize_label(y_test)
        if metrics_ordered is None: metrics_ordered = NNS.metrics_ordered # ['loss', 'acc', 'auc_roc']
        y_test = lb.transform(y_test)
        scores = trained_model.evaluate(X_test, y_test, verbose=0)
        print('lstmEvaluate> scores:\n%s\n' % zip(NNS.metrics_ordered, scores))  # loss, accuracy, auc 
        res = {} # output
        for i, metric in enumerate(metrics_ordered): 
            # metric = NNS.score_map[i]
            res[metric] = scores[i]   # use this to assess over/under fitting i.e gap between training and test errors
        return res
    def history_evluation(history, metrics_ordered=None, n_metrics=None):

        # observe model fitting quality (underfitting? overfitting? just right?)
        file_id = kargs.get('identifier', '')
        nlastm = 15
        for mt, mv in [('acc', 'val_acc'), ('loss', 'val_loss'), ]: 
            # [test]
            for m in (mt, mv): 
                print('... ... metric (%s): %s (n=%d)' % (m, history.history[m][-nlastm:], len(history.history[m])))
        
            # generalization error 
            # print('... ... metric (gap): %s' % history.history[mt][-nlastm:]-history.history[mv][-nlastm:])
            save_plot(plot_metric(train=mt, test=mv), identifier=file_id, metric=mt)

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

    import dnn_utils as dnn 
    from dnn_utils import NNS
    import plotUtils as putils
    from seqparams import Graphic
    import modelSelect as ms

    y_uniq = np.unique(y_test)
    n_classes = len(y_uniq)
    lb = binarize_label(y_uniq)

    if model is None: 
        model = load()  # params: identifier
    assert model is not None, "Null model!"

    # [design] leave out history? 
    # history = kargs.get('history', None) # training history obtained from calling model.fit()
    # if history is not None: 
    #     scores, gaps = history_evluation(history, metrics_ordered=NNS.metrics_ordered)  # ['loss','acc', 'auc_roc']

    # [note] metrics_ordered should conform to Kera's return value of model.evaluate
    res = {}
    res = general_evaluation(X_test, y_test, trained_model=model, lb=lb)  # params: metrics_ordered: ['loss', 'acc', 'auc_roc', ] by default
    res.update(roc_evaluation(X_test, y_test, trained_model=model))  # attributes: {min, max, micro, macro, loss, accuracy}

    ### model persistence 
    ref_performance = kargs.get('ref_performance', 0.5)  # only used in modelEvluateBatch()
    if kargs.get('save_model', True): 
        tSaveEff = False
        if kargs.get('conditional_save', True): 
            target_metric = kargs.get('target_metric', 'loss')  # this is not the same as the monintored metric above
            if target_metric.startswith('los'): 
                if res['loss'] <= ref_performance: tSaveEff = True
            else: 
                if res[target_metric] >= ref_performance: tSaveEff = True
        else: 
            tSaveEff = True

        if tSaveEff: 
            print('... saving a model with metric(%s): %f' % (target_metric, res[target_metric]))
            save(model)
    return res  # matric -> scores

# define roc_callback 
def auc_roc(y_true, y_pred):
    """

    Reference
    ---------
    https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
    """
    # any tensorflow metric
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

# example making new probability predictions for a classification problem
def demo_classify(): 
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.preprocessing import MinMaxScaler

    # generate 2d classification dataset
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
    scalar = MinMaxScaler()
    scalar.fit(X)
    X = scalar.transform(X)

    # define and fit the final model
    model = Sequential()
    model.add(Dense(4, input_dim=2, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(X, y, epochs=500, verbose=0)

    # new instances where we do not know the answer
    Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
    Xnew = scalar.transform(Xnew)
    
    # make a prediction
    ynew = model.predict_proba(Xnew)

    # show the inputs and predicted outputs
    for i in range(len(Xnew)):
	    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))

def t_read(**kargs): 
    from config import sys_config 
    import pandas as pd

    prefix = sys_config.read('ProjDir')
    suffix = "seqmaker/data/CKD/combined/tset-n0-IDregular-pv-dm2-regular-visit-tLb100-mLen10-GCKD.csv"
    fpath = os.path.join(prefix, suffix)
    assert os.path.exists(fpath), "Invalid input path:\n%s\n" % fpath

    chunksize = 1000
    for i, ts in enumerate(pd.read_csv(fpath, chunksize=chunksize, iterator=True)):
        print('[%d] dim(ts): %s' % (i, str(ts.shape)))

    chunksize = 5000
    for i, ts in enumerate(pd.read_csv(fpath, chunksize=chunksize, iterator=True)):
        if i < 3: print('... 2nd read> [%d] dim(ts): %s' % (i, str(ts.shape)))

    return

def test(**kargs): 

    # make probability predictions 
    # demo_classify()

    # misc tests 
    t_read()  # read (d2v) training data

    return 

if __name__ == "__main__": 
    test()