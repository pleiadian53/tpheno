import random, os
import numpy as np
from scipy import interp  # interperlation 
from scipy.stats import sem  # compute standard error
import collections

import matplotlib

# Generate images without having a window appear
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.

from matplotlib import pyplot
import matplotlib.pyplot as plt

from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
from utils import div

try:
    import cPickle as pickle
except:
    import pickle

# performance metrics 
from sklearn.metrics import roc_curve, auc, roc_auc_score

import configure 
DataRoot = configure.DataRoot
DataExpRoot = configure.DataExpRoot
ProjDir = configure.ProjDir
# LabName, LabTest = 'cerner'
DataExpDir = os.path.join(ProjDir, 'experiment')   # this can change 
DataDir = 'data-learner'

plt.rcParams['figure.figsize'] = 10, 7.5
plt.rcParams['axes.grid'] = True
plt.gray()

class Domain(object): 
    roc_prefix = 'roc'
    fsep = '-'

    models = {'antibio': 'antibiotic', 
              'microbio': 'microbiotic',
              'urine': 'unitary chemistry', 
              'blood': 'intravenous chemistry',
             }

    @staticmethod
    def nameROCPlot(**kargs): 
        """
        Name ROC plot based on the plot index. 


        Memo
        ----
        subsumed by the version in ROC
        naming convention 
           roc-008.61-antibio.png 
        """
        ext = kargs.get('ext', 'png')
        identifier = kargs.get('identifier', None)
        fsep = kargs.get('fsep', Domain.fsep)  # file separator
        level = kargs.get('level', None)
        if not identifier: 
            identifier = ''
            code = kargs.get('code', kargs.get('target', None))
            if code: identifier += code 
            group = kargs.get('group', kargs.get('labtest', None))
            # if level == 0: assert group is not None
            if group: identifier += fsep + group
            if not identifier:  # then must be global over all code and all models 
                identifier = 'global'
        return '%s-%s.%s' % (Domain.roc_prefix, identifier, ext)

class ROC(object): 
    """

    Note
    ----
    1. typical operation sequence, add -> eval -> get/select -> plot
    """
    cgmap = {}  # (level-0) index into code and group 

    stats_attr = ['mean', 'ci', ]
    # statsl0 = {k:{} for k in stats_attr}

    # cgmap_mean = {}  # (code, group) -> score
    # cgmap_ci = {}  # estimates of confidence intervals  (code, group) -> CI

    cmap = {}  # (level-1) index into code, ROC for ts2 (which combines all group models)
    gmap = {}  # (level-1) index into group  
    tmap = {}  # (level-2) the final global model

    stats = {}; 

    codes = {} # subsume codel1 ... 
    groups = {}
    nulls = {} 

    entry_level2 = 'global'

    # groupl1 = set([]) 

    level0_model_complete = False
    level1_model_complete = False
    level1a_model_complete = False
    level2_model_complete = False

    repr1D = repr2D = None  # not in use yet
    prefix = 'roc'
    fsep = '-'  # file name separator

    n_folds = 5
    folds = {}
    n_folds_l0 = 5
    n_folds_l1 = n_folds_1a = 5  # n-folds CV folds for the inner loop (using h1- to produce ts2 and learn per-disease h2)
    n_folds_l2 = 5

    prefix = 'roc-stats'

    # 2, 2.1 => 3, 3.1 
    levels = [0, 1, 1.5, 2, 2.1, 2.5, 2.6, 2.7, 2.8, 3, 3.1]  
    levels.extend(['1a', '1b', '1c', '1t'])  # 2.5, 2.6, 2.7, 1.5

    # models = configure.Params.bt_models

    @staticmethod
    def init(**kargs): 
        ROC.cgmap = {}  # map from (code, model) to cvset @ level 0
        ROC.cmap, ROC.gmap, ROC.tmap = {}, {}, {}  # level-1 and level-2 map

        for level in ROC.levels: 
            ROC.stats[level] = ROC.stats_struct()

        for level in ROC.levels: 
            ROC.codes[level] = set()

        ROC.groups[0] = set()

        # ROC.codel0, ROC.groupl0 = set([]), set([])
        # ROC.codel1 = set([])
        # ROC.codel1a = set()
        # ROC.codel2 = set([])   # level 2 contains subset of codes from level 1 that are annotated as gold standard data

        for level in ROC.levels: 
            ROC.nulls[level] = []

        # ROC.null_level0, ROC.null_level1, ROC.null_level1a = [], [], []
        ROC.reset_status() 

        ROC.n_folds = ROC.n_folds_l0 = ROC.n_folds_l1 = ROC.n_folds_1a = ROC.n_folds_l2 = 5

        for level in ROC.levels: 
            ROC.folds[level] = 5
        print('info> ROC.init complete.')
        return 

    @staticmethod
    def reset(**kargs):
        ROC.init(**kargs)
        return 

    @staticmethod
    def save(path=None, fname=None, level=1, **kargs):
        if path is None: path = 'data-learner'
        assert os.path.exists(path), "ROC.save> data file root path does not exist: %s" % path

        print('ROC.save> size of entries: %d' % ROC.size(level=level))

        data = {}
        data['cgmap'] = ROC.cgmap
        data['cmap'], data['gmap'], data['tmap'] = (ROC.cmap, ROC.gmap, ROC.tmap)
        data['stats'] = ROC.stats

        # data['codel0'], data['groupl0'] = ROC.codes[0], ROC.groups[0]
        # data['codel1'] = ROC.codel1
        # data['codel1a'] = ROC.codel1a
        # data['codel2'] = ROC.codel2

        data['codes'] = ROC.codes
        data['groups'] = ROC.groups
        data['nulls'] = ROC.nulls
        data['folds'] = ROC.folds

        data['null_level0'], data['null_level1'] = ROC.nulls[0], ROC.nulls[1]
        
        data['n_folds'] = ROC.n_folds
        data['n_folds_l0'] = ROC.n_folds_l0
        data['n_folds_l1'] = ROC.n_folds_l1
        data['n_folds_1a'] = ROC.n_folds_1a
        data['n_folds_l2'] = ROC.n_folds_l2

        # save states also? 
        # data['l0_model_complete'] = ROC.l0_model_complete

        if fname is None: fname = ROC.nameDataFile(level=level, **kargs)
        fp = os.path.join(path, fname)
        print('ROC.save> saving ROC data to %s' % fp)
        pickle.dump(data, open(fp, "wb" ))

        return

    @staticmethod
    def load(path=None, fname=None, level=1, **kargs): 
        if path is None: path = 'data-learner'
        if fname is None: fname = ROC.nameDataFile(level=level, **kargs)
        fp = os.path.join(path, fname)
        if not os.path.exists(fp): 
            print("ROC.load> ROC stats file does not exist at: %s" % fp)
            return False
        print('ROC.load> loading ROC data from %s' % fp)
        data = pickle.load(open(fp, "rb" ))
        if not data: 
            print('ROC.load> Null stats data.')
            return False

        # unpack 
        ROC.cgmap = data.get('cgmap', {})
        ROC.cmap, ROC.gmap, ROC.tmap = data.get('cmap', {}), data.get('gmap', {}), data.get('tmap', {})
        ROC.stats = data.get('stats', {})
        if not ROC.stats: 
            for level in ROC.levels: 
                ROC.stats[level] = ROC.stats_struct()

        ROC.codes = data.get('codes', {})
        if not ROC.codes: 
            for level in ROC.levels: 
                 ROC.codes[level] = set() 
        ROC.groups = data.get('groups', {})
        if not ROC.groups: 
            ROC.groups[0] = set()
        # ROC.codel0, ROC.groupl0 = data.get('codel0', set()), data.get('groupl0', set())
        # ROC.codel1 = data.get('codel1', set())
        # ROC.codel1a = data.get('codell1a', set())
        # ROC.codel2 = data.get('codel2', set())

        ROC.nulls = data.get('nulls', {})
        if not ROC.nulls: 
            for level in ROC.levels: 
                ROC.nulls[level] = []
        ROC.folds = data.get('folds', {})
        # ROC.null_level0, ROC.null_level1 = data.get('null_level0', []), data.get('null_level1', [])
        # ROC.null_level1a = data.get('null_level1a', [])

        ROC.n_folds = data.get('n_folds', ROC.n_folds) 
        ROC.n_folds_l0 = data.get('n_folds_l0', ROC.n_folds_l0)
        ROC.n_folds_l1 = data.get('n_folds_l1', ROC.n_folds_l1)
        ROC.n_folds_1a = data.get('n_folds_1a', ROC.n_folds_1a)
        ROC.n_folds_l2 = data.get('n_folds_l2', ROC.n_folds_l2)

        print('ROC.load> size of entries: %d' % ROC.size(level=level))

        return True

    @staticmethod
    def size(level=0, type_='code'):
        if level == 0: 
            return len(ROC.cgmap)
        if level == 1: 
            if type_.startswith('c'): 
                return len(ROC.cmap) 
            else: 
                return len(ROC.gmap)
        if level == 2: 
            return len(ROC.tmap)
        print('ROC.size> Unknown level %s' % level)
        return 0

    @staticmethod
    def nameDataFile(level=1, **kargs): 
        """

        Related
        -------
        nameROCPlot()

        Note
        ----
        1. order: 
           prefix, level, identifier, suffix, meta

        """
        prefix = kargs.get('prefix', ROC.prefix)
        fname = prefix
        lstr = 'l%s' % level
        fname = fname + '-' + lstr  # [protocol] '-' is not the fsep '_' is 
        identifier = kargs.get('identifier', None)
        if identifier is not None: 
            fname += '_' + identifier
        
        suffix = kargs.get('suffix', None)
        if suffix is not None: fname += '_' + suffix
        meta = kargs.get('meta', None)
        if meta is not None: fname += '_' + meta

        ext = kargs.get('ext', 'pkl')
        return fname + '.' + ext

    @staticmethod
    def stats_struct(): 
        return {k:{} for k in ROC.stats_attr}

    @staticmethod
    def reset_status(): 
        ROC.level0_model_complete = False
        ROC.level1_model_complete = False
        ROC.level2_model_complete = False
        ROC.repr1D = ROC.repr2D = None

    @staticmethod
    def set_cv(n_folds, level=1, **kargs):  # [todo] generalize this to high # of layers? 
        # assert level in (0, 1, 2, 1.5, 2.5, 3, )
        if n_folds is None: 
            print('ROC.set_cv> use default.')
            return 
        meta = kargs.get('message_', None)
        msg = 'ROC.set_cv> setting n_folds to %d, at level %d' % (n_folds, level)
        msg = (msg + ' (%s)\n' % meta) if meta is not None else (msg + '\n')
        print msg

        ROC.folds[level] = n_folds
        # if level == 0: 
        #     ROC.folds[0] = n_folds
        # elif level in (1, 1.5, ): 
        #     ROC.folds[1] = n_folds
        # elif level in (2, 2.5, ): 
        #     ROC.folds[2] = n_folds
        return 
    @staticmethod
    def getNFold(level):
        return ROC.folds[level]

    @staticmethod
    def show_cv(**kargs): 
        for level, n_folds in ROC.folds.items(): 
            print('ROC.show_cv> level: %s, n_folds: %d' % (level, n_folds))
        return

    @staticmethod
    def nameROCPlot(**kargs): 
        """
        Name ROC plot based on the plot index. 


        Memo
        ----
        naming convention 
           roc-008.61-antibio.png 
        """
        def get_name(alist):
            return fsep.join([e for e in alist if e]) 

        ext = kargs.get('ext', 'pdf')
        identifier = kargs.get('identifier', None)
        fsep = kargs.get('fsep', ROC.fsep)  # file separator
        level = kargs.get('level', None)
        code = group = ''
        meta = kargs.get('meta', None)

        # figure out code and group if possible
        if True: 
            entry = kargs.get('entry', None)
            if entry: # order: code, name, meta (for extra info)
                if len(entry) >= 2: 
                    level = 0
                    code, group = entry
                    meta = entry[2:]
                elif hasattr(entry, "__iter__") and len(entry) == 1: 
                    code = entry[0]; 
                    assert not level or level == 1
                elif isinstance(entry, str) and not entry != ROC.entry_level2: 
                    try: 
                        float(entry)
                    except: 
                        raise ValueError, 'ROC.nameROCPlot> Invalid name: %s' % entry
                    code = entry; assert not level or level == 1
                else: 
                    assert not level or level == 2
            else: 
                code = kargs.get('code', kargs.get('target', None))
                group = kargs.get('group', kargs.get('labtest', None))
                if not level: 
                    if code is not None and group is not None: 
                        level = 0
                
        if not identifier: 
            identifier = get_name([code, group])
            if not identifier:  # then must be global over all code and all models 
                if level: 
                    identifier = 'roc_level%d' % level
                else: 
                    identifier = 'roc_test'
        else: 
            if code and identifier.find(code) < 0: 
                identifier += fsep + code 
            if group and identifier.find(group) < 0: 
                identifier += fsep + group
            if level is not None: 
                level = 'l%s' % level
                if identifier.find(level) < 0: 
                    identifier += fsep + level

        if meta: # user notes
            # is meta already part of the identifier? 
            if identifier.find(meta) < 0: 
                identifier += fsep + meta
        return '%s-%s.%s' % (ROC.prefix, identifier, ext)

    @staticmethod
    def evall0(reset=True): 
        """
        Find all the codes and names of group models. 
        """
        if not reset and ROC.level0_model_complete: 
            return (len(ROC.codes[0]), len(ROC.groups[0]))
        for k, _ in cgmap.items(): 
            c, gn = k
            ROC.codes[0].update( [c] )
            ROC.groups[0].update( [gn] )
        ROC.level0_model_complete = True 
        return (len(ROC.codes[0]), len(ROC.groups[0]))

    @staticmethod
    def evall1(reset=False): 
        if not reset and ROC.level1_model_complete: return len(ROC.codes[1])
        ROC.codes[1] = set(cmap)  # by default set contains keys

        if ROC.level0_model_complete: 
            n0, n1 = len(ROC.codes[0]), len(ROC.codes[1])
            assert n1 >= n0, "Level1 has fewer models! l0:%d > l1:%d" % (n0, n1)
            if n1 > n0: 
                div(message="Some codes are missing base models: Level0: %d codes < Level1: %d" % (n0, n1))
        ROC.level1_model_complete = True
        return len(ROC.codes[1])  

    @staticmethod
    def evall2(reset=False):
        if not reset and ROC.level2_model_complete: return len(ROC.codes[2])
        ROC.codes[2] = set(tmap)
        if ROC.level1_model_complete: 
            n1, n2 = len(ROC.codes[1]), len(ROC.codes[2])
            assert n1 >= n2, "Level2 has more codes? Impossible! n1: %d < n2: %d" % (n1, n2)
        ROC.level2_model_complete = True
        return 

    @staticmethod
    def add(code, group=None, cvset=None, level=0, **kargs): 
        if not cvset: raise ValueError, "Missing (fpr,tpr) estimates."
        if level == 0: 
            return ROC.addl0(code=code, group=group, cvset=cvset, **kargs)
        elif level == 1: 
            return ROC.addl1(code=code, cvset=cvset, **kargs)  # for now, assume adding new code
        return ROC.addl2(cvset, code=code, **kargs)

    @staticmethod
    def addl0(code, group, cvset, **kargs): 
        """
        Add (fpr,tpr)s to level0 plot. 
        """
        if kargs.has_key('n_folds'): 
            ROC.folds[0] = kargs['n_folds']
        try: 
            float(code)
        except: 
            raise ValueError, 'ROC::add> Possibly not valid code: %s' % code
        assert hasattr(cvset, '__iter__')
        assert len(cvset) == ROC.folds[0], "Number of (fpr,tpr)s not equal to n_folds: %d != %d?" % \
                                            (len(cvset), ROC.folds[0])
        k = (code, group)
        n_antes = len(ROC.cgmap)
        ROC.cgmap[k] = cvset  # [(), (), ()] where () contains fpr,tpr 
        print('ROC.addl0> added cvset of size %d to cgmap of size: %d <- %d' % (len(cvset), len(ROC.cgmap), n_antes))
        ROC.codes[0].update([code])
        ROC.groups[0].update([group])
        return 
    @staticmethod
    def getl0(codes, groups=None, **kargs):
        print('ROC.getl0> Input %d codes and %d groups.' % (len(codes), len(groups)))
        if not ROC.codes[0]: ROC.evall0(reset=True)
        codes = [code for code in codes if code in ROC.codes[0]]
        groups = [g for g in groups if g in ROC.groups[0]]
        print('ROC.getl0> valid: %d codes and %d groups.' % (len(codes), len(groups)))   
        return ROC.selectl0(codes, groups=groups, dim=2, **kargs)    
 
    @staticmethod
    def addl1(code, cvset, group=None, **kargs): 
        """
        Add (fpr,tpr)s to level1 plot. 
        """
        if kargs.has_key('n_folds'): 
            ROC.folds[1] = kargs['n_folds']
        try: 
            float(code)
        except: 
            if not code.startswith( ('com', 'glo') ): 
                raise ValueError, 'ROC::addl1> Possibly not valid code: %s' % code
            else: 
                print('addl1> Assuming that the code represents a global model at level 1.')
        assert hasattr(cvset, '__iter__') 
        assert len(cvset) == ROC.folds[1], "Number of (fpr,tpr)s not equal to n_folds: %d != %d?" % \
                                            (len(cvset), ROC.folds[1])
        ROC.cmap[code] = cvset   
        ROC.codes[1].update([code])
        return
    @staticmethod
    def getl1(codes, **kargs):  # get is the same as select
        pass 

    @staticmethod
    def addl2(cvset, code=None, **kargs):
        if kargs.has_key('n_folds'): 
            ROC.folds[2] = kargs['n_folds']
        assert hasattr(cvset, '__iter__')
        if code is None: 
            code = kargs.get('key_', ROC.entry_level2)  # global, code-agnostic model
        assert len(cvset) == ROC.folds[2], "Number of (fpr,tpr)s not equal to n_folds: %d != %d?" % \
                                            (len(cvset), ROC.folds[2])
        ROC.tmap[code] = cvset
        ROC.codes[2].update([code])
        return 
    @staticmethod
    def getl2(codes=None, **kargs):
        pass

    @staticmethod
    def select(codes=None, dim=2, level=1, **kargs): 
        cvset = None
        if level == 0: 
            assert codes is not None
            groups = kargs.get('groups', ROC.groups[0])
            cvset = ROC.selectl0(codes, groups=groups, dim=dim, **kargs)
        elif level == 1: 
            assert codes is not None
            cvset = ROC.selectl1(codes=codes, dim=dim, **kargs)
        elif level == 2: 
            cvset = ROC.selectl2(codes=codes, dim=dim, **kargs)

        if not cvset: 
            raise ValueError, "ROC.select> Possibly unknown level: %s or no data available for codes: %s" % \
                    (level, str(codes))
        return cvset

    @staticmethod
    def selectl0(codes, groups=None, dim=2, **kargs):
        if not groups: groups = ROC.groups[0]
        if dim == 1: # 1D plot, everything in one row
            return ROC.assemble1D(codes=codes, groups=groups, level=0, **kargs)
        return ROC.assmeble2D(codes=codes, groups=groups, level=0, **kargs)

    @staticmethod
    def selectl1(codes, dim=2, **kargs):
        """

        Note
        ----
        1. at level1, group models are marginalized 
        """
        # if not codes: codes = ROC.codes[0]
        if dim == 1: # 1D plot, everything in one row
            return ROC.assemble1D(codes=codes, level=1, **kargs)
        return ROC.assmeble2D(codes=codes, level=1, **kargs)  

    @staticmethod
    def selectl2(codes=None, dim=2, **kargs): 
        if codes is None: codes = [ROC.entry_level2, ]
        if dim == 1: # 1D plot, everything in one row
            return ROC.assemble1D(codes=codes, level=2, **kargs)
        return ROC.assmeble2D(codes=codes, level=2, **kargs)       
        
    @staticmethod
    def iterl0(codes=None, groups=None, **kargs):
        if not ROC.codes[0]: ROC.evall0()
        assert len(ROC.codes[0]) > 0
        if not groups: 
            groups = ROC.groups[0]
        else: 
            groups = [g for g in groups if g in ROC.groups[0]]

        if not codes: 
            codes = ROC.codes[0] 
        else: 
            n_total, n_eff = len(codes), 0
            codes = [c for c in codes if c in ROC.codes[0]]
            n_eff = len(codes)
            print('ROC.iterl0> after filtering %d -> %d' % (n_total, n_eff))

        for code in codes: 
            for group in groups: 
                k = (code, group)
                if not (k in ROC.nulls[0]): 
                    yield (code, group)
                    
    @staticmethod
    def iterl1(codes=None, **kargs):
        if not ROC.codes[1]: ROC.evall1()
        assert len(ROC.codes[1]) > 0
        if not codes: 
            codes = ROC.codes[1]
        else: 
            n_total, n_eff = len(codes), 0
            codes = [c for c in codes if (c in ROC.codes[1]) and (not c in ROC.nulls[1])]
            n_eff = len(codes)
            print('ROC.iterl1> after filtering %d -> %d' % (n_total, n_eff))

        for code in codes: 
            yield code  

    @staticmethod
    def iterl2(codes=None, **kargs):
        if not ROC.codes[2]: ROC.evall2()
        assert len(ROC.codes[2]) > 0 
        if not codes: 
            codes = ROC.codes[2]
        else:
            n_total, n_eff = len(codes), 0
            codes = [c for c in codes if (c in ROC.codes[2]) and (not c in ROC.nulls[2])]
            n_eff = len(codes)
            print('ROC.iterl2> after filtering %d -> %d' % (n_total, n_eff)) 

        for code in codes: 
            yield code

    @staticmethod
    def assemble1D(codes, groups=None, level=0, **kargs):
        cvset = []
        # if level == 0: 
        #     iter_ = ROC.iterl0; map_ = ROC.cgmap
        # elif level == 1: 
        #     iter_ = ROC.iterl1 map_ = ROC.cmap

        # for k in iter_(codes=codes, groups=groups): 
        #     cvset.append(map_[k])  # [ [(),()], [(),()] ]
        if level == 0: 
            if not groups: groups = ROC.groups[0]
            for k in ROC.iterl0(codes=codes, groups=groups): 
                cvset.append(ROC.cgmap[k])
        elif level == 1: 
            for k in ROC.iterl1(codes=codes): 
                cvset.append(ROC.cmap[k])
        elif level == 2: 
            for k in ROC.iterl2(codes=codes): 
                cvset.append(ROC.cmap[k])

        return cvset

    @staticmethod
    def assmeble2D(codes, groups=None, level=0, **kargs):
        def v_fold(cvset):
            msg = "assemble2D.v_fold> Ill repr: n_folds %d while detected %d" % (n_folds, len(cvset))
            # assert len(cvset) == n_folds, msg
            if len(cvset) != n_folds: 
                print msg
                ROC.show_cv()

        ncol = kargs.get('ncol', 4)   # number of columns of ROC plots in a row
        n_folds = kargs.get('n_folds', ROC.getNFold(level))
        print('ROC.assemble2D> ncol: %d, n_fold: %d' % (ncol, n_folds)) 
        cvset2D, cvset = [], []
        
        if level == 0: 
            if not groups: groups = ROC.groups[0]
            cvset = None
            for i, k in enumerate(ROC.iterl0(codes=codes, groups=groups)): 
                if i % ncol == 0: 
                    if i > 0: cvset2D.append(cvset)  # copy and add
                    cvset = []
                v_fold(ROC.cgmap[k])
                cvset.append(ROC.cgmap[k])
        elif level == 1: 
            # print('ROC.assmeble2D> n_codes: %d vs valid codes: %d' % (len(codes), len([c for c in ROC.iterl1(codes=codes)])))
            for i, k in enumerate(ROC.iterl1(codes=codes)): # foreach code
                if i % ncol == 0: 
                    if i > 0: 
                        # assert len(cvset) == ncol
                        cvset2D.append(cvset)  # one row
                        cvset = [] 
                v_fold(ROC.cmap[k])
                cvset.append(ROC.cmap[k])
        elif level == 2:  # may need to use leave one out CV
            # print('ROC.assmeble2D> n_codes: %d vs valid codes: %d' % (len(codes), len([c for c in ROC.iterl2(codes=codes)])))
            for i, k in enumerate(ROC.iterl2(codes=codes)): 
                if i % ncol == 0: 
                    if i > 0: 
                        # assert len(cvset) == ncol
                        cvset2D.append(cvset)  # one row
                        cvset = [] 
                v_fold(ROC.cmap[k])
                cvset.append(ROC.cmap[k])
            
        if cvset: # collect the last piese
            v_fold(cvset[0]); assert len(cvset) == ncol
            cvset2D.append(cvset)
        
        # final test
        if (level == 0 and ncol == len(groups)) or (level==1): 
            assert len(cvset2D)*ncol == len(codes), "Ill repr: size (%d vs %d) cvset2D:\n%s\n" % (len(cvset2D), len(codes), cvset2D)

        return cvset2D

    @staticmethod
    def is_plot_element(e, n_folds=None):
        import random
        if not hasattr(e, '__iter__'): return False 
        rocelem = None
        if n_folds is None: n_folds = ROC.n_folds
        ncv = len(e)  # n graphs, each using n fold CV
        assert ncv == n_folds, 'ROC.is_plot_element> expected n_folds=%d but got %d' % (n_folds, ncv)

        try: 
            rocelem = e[random.randint(0, ncv-1)]  
        except: 
            print('is_plot_element> not a legal plot element: %s' % rocelem)
            return False 
        if not hasattr(rocelem, '__iter__'): return False 
        return len(rocelem) >= 2

    @staticmethod
    def verify1D(cvset, n_folds=None):
        import random
        rowplt = cvset
        print('ROC.verify1D> number of graphs in a row ~ ncol ~ %d' % len(rowplt))
        e = random.randint(0, len(rowplt)-1)
        if n_folds is None: n_folds = ROC.n_folds
        assert ROC.is_plot_element(rowplt[e], n_folds=n_folds), "ROC.verify1D> Invalid repr: %s" % rowplt[e] 

    @staticmethod
    def verify2D(cvset2D, ncol=4, n_folds=None):
        import random
        e = random.randint(0, len(cvset2D)-1)
        ROC.verify1D(cvset2D[e], n_folds=n_folds)
            
        nr = len(cvset2D)
        nc = len(cvset2D[0]) 
        assert nr >=1 and nc == ncol 
        return

    @staticmethod
    def verify(level=0, n_folds=None):
        def v_cvset(cvset, n_folds=None):
            if n_folds is None: n_folds = ROC.n_folds
            assert len(cvset) == n_folds
            sizes = [len(e) for e in cvset]
            n = sizes[0] 
            assert n in (2, 3)
            assert sum([1 for x in sizes[1:] if (x != n)]) == 0, "ROC.verify> inconsistent sizes: %s" % sizes 

        import predicate
        if level == 0: 
            if n_folds is None: n_folds = ROC.folds[0]
            for k, v in ROC.cgmap.items(): 
                assert len(k) == 2
                c, m = k
                assert predicate.isDiagnosticCode(c)
                v_cvset(v, n_folds=n_folds)
        elif level == 1: 
            if n_folds is None: n_folds = ROC.folds[1]
            for k, v in ROC.cmap.items(): 
                if not predicate.isDiagnosticCode(k): 
                    print("ROC.verify> %s is a group name?" % k)
                v_cvset(v, n_folds=n_folds)
        elif level == 2: 
            if n_folds is None: n_folds = ROC.folds[2]
            for k, v in ROC.tmap.items(): 
                if not k == ROC.entry_level2: 
                    print("ROC.verify> possibly invalid level-2 key: %s" % k)
                v_cvset(v, n_folds=n_folds)
        else: 
            print('ROC.verify> level: %s, no data collected for now.' % level)
        return

    @staticmethod
    def executel0(**kargs):
        """
        Infer AUC and mean AUC from the CV data (i.e. (fpr,tpr)-set). 
        """
        scores = [] 
        assert len(ROC.cgmap) > 0, "ROC.executel0> cgmap has no data!"
        for k, cvset in ROC.cgmap.items(): 
            ncv = len(cvset)
            interp_err = False
            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            for i, e in enumerate(cvset): 
                fpr, tpr = e[0], e[1]
                try: 
                    mean_tpr += interp(mean_fpr, fpr, tpr)
                    mean_tpr[0] = 0.0
                except Exception, e:
                    print("ROC.executel0> Skipping case %s due to: %s" % (str(k), e)) 
                    ROC.nulls[0].append(k)
                    interp_err = True
                    break
                # roc_auc = e[2] if len(e) >= 2 else None
                # if not roc_auc: 
                roc_auc = auc(fpr, tpr)   
                scores.append(roc_auc)
            
            if not interp_err: 
                mean_tpr /= ncv
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                ROC.stats[0]['mean'][k] = mean_auc
        return

    @staticmethod
    def rankl0(codes=None, groups=None, average=False, **kargs):
        """

        Arguments
        ---------
        average: if True, then take average over all group models and rank codes 
                 according to the averaged AUC. 

        Todo
        ----
        1. show also ROC.stats[level]['ci']
        """
        # import copy
        if not ROC.stats[0]['mean']: ROC.executel0(**kargs) 

        # [test]
        print("info> ROC.stats[0]['mean']:\n%s\n" % ROC.stats[0]['mean'])

        tb = {}
        if codes or groups: 
            for k, v in ROC.stats[0]['mean'].items(): 
                c, gn = k
                if (codes and not c in codes) or (groups and not gn in groups): 
                    continue
                tb[k] = v
            rcgv = sorted([(k, v) for k, v in tb.items()], key=lambda e:e[1], reverse=True) # high -> low
        else: 
            # tb = copy.deepcopy(ROC.statsl0['mean'])
            print('ROC.rankl0> total of %d code+group combos of means' % len(ROC.stats[0]['mean']))

            # tb = {}
            # rank wrt to code only (take the average of group)
            # for k, v in ROC.statsl0['mean'].items(): 
            #     c, m = k
            #     if not tb.has_key(c): tb[c] = []
            #     tb[c].append(v)
            # for c, scores in tb.items(): 
            #     tb[c] = np.mean(scores)  # or max(scores)

            rcgv = sorted([(k, v) for k, v in ROC.stats[0]['mean'].items()], key=lambda e:e[1], reverse=True) # high -> low
                    
        # rcgv = sorted([(k, v) for k, v in tb.items()], key=lambda e:e[1], reverse=True) # high -> low
        head, tail = kargs.get('head', kargs.get('n_head', None)), kargs.get('tail', kargs.get('n_tail', None))
        if head: 
            return rcgv[:head]
        if tail: 
            return rcgv[-tail:]

        for index, mean in rcgv: 
            print('ROC.rankl0> index: %s, mean: %f, CI: %s' % (str(index), mean, str(ROC.stats[0]['ci'][index])) )
        return rcgv 

    @staticmethod
    def execute(level=1, **kargs): 
        if level == 0: 
            return ROC.executel0(**kargs)
        
        M = Nl = S = None
        if level == 1: 
            M = ROC.cmap
            Nl = ROC.nulls[1]
            S = ROC.stats[1] # ROC.statsl1
        elif level == 2: 
            M = ROC.tmap 
            Nl = ROC.nulls[2]
            S = ROC.stats[2] # ROC.statsl2

        scores = []
        for k, cvset in M.items(): 
            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            nfolds = len(cvset)
            # print('executel1-test> %d (fpr,tpr)s.' % ncv)
            interp_err = False

            for i, e in enumerate(cvset): 
                fpr, tpr = e[0], e[1]

                try: 
                    mean_tpr += interp(mean_fpr, fpr, tpr)
                    mean_tpr[0] = 0.0
                except Exception, e:
                    print("ROC.execute> Level:%d, skipping case %s due to: %s" % (level, str(k), e)) 
                    Nl.append(k)
                    interp_err = True
                    break 
                # roc_auc = e[2] if len(e) >= 2 else None
                # if not roc_auc: 
                roc_auc = auc(fpr, tpr)  
                scores.append(roc_auc) 
            
            if not interp_err: 
                mean_tpr /= nfolds
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                S['mean'][k] = mean_auc 
        return S 

    @staticmethod
    def executel2(**kargs):
        return ROC.execute(level=2, **kargs) 

    @staticmethod
    def executel1(**kargs): 
        return ROC.execute(level=1, **kargs)

    @staticmethod
    def rank(codes=None, groups=None, level=1, **kargs):
        if level == 0: 
            return ROC.rankl0(codes=codes, groups=None, **kargs)
        if level == 1: 
            M = ROC.stats[1]['mean']
        elif level == 2: 
            M = ROC.stats[2]['mean']

        if not M: 
            M = ROC.execute(level=level, **kargs) 

        if codes: 
            statstb = {code: M[code] for code in codes if code in M.keys()}
        else: 
            statstb = M
        rcgv = sorted([(k, v) for k, v in statstb.items()], key=lambda e:e[1], reverse=True) # high -> low
        head, tail = kargs.get('head', None), kargs.get('tail', None)
        if head: 
            return rcgv[:head]
        if tail: 
            return rcgv[-tail:]

        div(message='ROC.rank> Ranking in terms of means at level %d ...' % level)
        for index, mean in rcgv: 
            print('ROC.rank> index: %s, mean: %f, CI: %s' % (str(index), mean, str(ROC.stats[level]['ci'][index])) )            
        return rcgv

    @staticmethod
    def rankl1(codes=None, **kargs): 
        if not ROC.stats[1]['mean']: ROC.executel1(**kargs) 
        M = ROC.stats[1]['mean'] # ROC.statsl1['mean']

        pairs = None
        if codes: 
            pairs = [(c, M[c]) for c in codes if c in M] 
            assert len(codes) == len(pairs), "ROC.rank1> Invalid codes exist in input:\n%s\n" % codes
        else: 
            pairs = [(c, v) for c, v in M.items()]
        rcgv = sorted(pairs, key=lambda e:e[1], reverse=True) # high -> low
        head, tail = kargs.get('head', None), kargs.get('tail', None)
        if head: 
            return rcgv[:head]
        if tail: 
            return rcgv[-tail:]

        div(message='ROC.rankl1> Ranking codes in terms of their means and show CIs ...')
        for index, mean in rcgv: 
            print('ROC.rankl1> index: %s, mean: %f, CI: %s' % (str(index), mean, str(ROC.stats[1]['ci'][index])) )

        return rcgv  

    @staticmethod
    def rankl2(codes=None, **kargs): 
        return ROC.rank(codes=codes, level=2, **kargs)                  

    @staticmethod
    def plot(cvset=None, clear_plot=True, **kargs):
        path = kargs.get('path', DataDir) 
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        # aucs = []
        # ax = kargs.get('ax', None)
        identifier = kargs.get('identifier', None)
        fname = kargs.get('fname', None)
        
        if cvset is None: cvset = ROC.tmap[ROC.entry_level2]
        assert len(cvset) > 0, "ROC.plot> Null (fpr,tpr) vectors. Nothing to plot."
        if clear_plot: plt.clf()

        # fig, ax = ax.subplots(1, n_groups)   # 1 row of n_group plots
        scores = []
        n_folds_eff = len(cvset)
        for i, e in enumerate(cvset):  # foreach cv result where each result has a (fpr, tpr)
            fpr, tpr = e[0], e[1]
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)

            if n_folds_eff <= 5: 
                plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % ( (i+1), roc_auc))
            else: 
                if i % 2 == 0: 
                    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % ( (i+1), roc_auc))

            # aucs.append(roc_auc)

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        mean_tpr /= len(cvset)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (%s)' % identifier)
        plt.legend(loc="lower right")

        # save 
        if fname is None: fname = ROC.nameROCPlot(**kargs) 
        if kargs.get('save_', True): 
            opath = os.path.join(path, fname)
            print('ROC.plot> saving plot to %s' % opath)
            plt.savefig(opath, bbox_inches='tight')  # [2]
        elif show_: 
            plt.show()

        return

    @staticmethod
    def getId(**kargs):
        entry = kargs.get('entry', None)
        if entry: return entry 
        code, group = kargs.get('code', None), kargs.get('group', None)
        if code and group: 
            return (code, group)
        if code: 
            return code
        return ROC.entry_level2

    @staticmethod
    def getCVSets(level=1, **kargs): 
        """
        Get (fpr,tpr)s by index. 
        """
        print("ROC.getCVSets> index: %s" % ROC.getId(**kargs))
        if level == 0:
            cvsets = ROC.cgmap[ROC.getId(**kargs)] 
        elif level == 1: 
             cvsets = ROC.cmap[ROC.getId(**kargs)]   # default level 1
        elif level == 2: 
            cvsets = ROC.tmap[ROC.getId(**kargs)]
        return cvsets

    @staticmethod
    def plot1D(codes=None, clear_plot=True, level=0, **kargs): 
        """
        Plot ROC curves in a row. 

        Use
        --- 
        1. Given a code, plot all its phenotypic models
        2. Plot several codes. 

        Note
        ----
        1. for level 0 data, select all the groups by default
        """
        cvsets = ROC.select(codes=codes, dim=1, level=level, **kargs)  # [1]
        n_folds = kargs.get('n_folds', Group.n_folds)
        if kargs.get('verify_', True): ROC.verify1D(cvsets, n_folds=n_folds)
        path = kargs.get('path', DataDir)
        identifier = kargs.get('identifier', None)
        title = kargs.get('title', 'Receiver Operating Characteristic')
        if cvsets is None: cvsets = ROC.getCVSets(level=level, **kargs)
        
        ncol = kargs.get('ncol', len(cvsets))

        # n_groups = len(cvsets)  # each group model has a cvset
        if clear_plot: plt.clf()

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        
        # a row of n_groups plots sharing same y axis
        fig, ax = plt.subplots(1, ncol, sharey=True)   

        # for i, a in enumerate(ax.flatten()):
        # scores = []
        for i, cvset in enumerate(cvsets):   # foreach model
            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            ncv = len(cvset)
            for j, e in enumerate(cvset): # foreach fold
                fpr, tpr = e[0], e[1]
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                ax[i].plot(fpr, tpr, lw=1)  # label='ROC fold %d (area = %0.2f)' % ( (i+1), roc_auc)

            ax[i].plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
            mean_tpr /= ncv
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            ax[i].plot(mean_fpr, mean_tpr, 'k--', 
                        label='Mean ROC (area = %0.2f)' % mean_auc, 
                        lw=2) # label='Mean ROC (area = %0.2f)' % mean_auc,
            ax[i].set_xlim([-0.05, 1.05])
            ax[i].set_ylim([-0.05, 1.05])
            # end each plot 

        # set common title and labels
        # fig.text(0.5, 0.04, 'False Positive Rate', ha='center')
        ax[0].set_ylabel('True Positive Rate')
        fig.text(0.04, 0.5, 'True Positive Rate', va='center', rotation='vertical')

        fig.suptitle(title, fontsize=12)
        # ax.set_xlabel('False Positive Rate')
        # ax.set_ylabel('True Positive Rate')

        # fig.tight_layout()

        # save 
        if fname is None: fname = Domain.nameROCPlot(**kargs) 
        if kargs.get('save_', True): 
            opath = os.path.join(path, fname)
            print('ROC.plot1D> saving plot to %s' % opath)
            plt.savefig(opath, bbox_inches='tight')  # [2]
        elif show_: 
            plt.show()

        return 

    @staticmethod
    def plot2D(codes, level=1, clear_plot=True, **kargs): 
        cvset2D = ROC.select(codes=codes, dim=2, level=level, **kargs)

        n_folds = kargs.get('n_folds', ROC.getNFold(level))
        if kargs.get('verify_', True): ROC.verify2D(cvset2D, n_folds=n_folds)
        path = kargs.get('path', DataDir)
        identifier = kargs.get('identifier', None) 
        title = kargs.get('title', 'Receiver Operating Characteristic')
        fname = kargs.get('fname', None)

        n_codes = len(cvset2D)   
        ncol = len(cvset2D[0])  # e.g. each group model has a cvset

        if clear_plot: plt.clf()

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        
        fig, ax = plt.subplots(n_codes, ncol, sharex=True, sharey=True)   

        # for i, a in enumerate(ax.flatten()):
        # scores = []
        for c, cvsets in enumerate(cvset2D): # foreach code 
            for i, cvset in enumerate(cvsets):   # foreach model
                mean_tpr = 0.0
                mean_fpr = np.linspace(0, 1, 100)
                ncv = len(cvset)
                for j, e in enumerate(cvset): # foreach fold
                    fpr, tpr = e[0], e[1]
                    mean_tpr += interp(mean_fpr, fpr, tpr)
                    mean_tpr[0] = 0.0
                    roc_auc = auc(fpr, tpr)
                    ax[c,i].plot(fpr, tpr, lw=1)  # label='ROC fold %d (area = %0.2f)' % ( (i+1), roc_auc)

                ax[c,i].plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  # diagnoal
                mean_tpr /= ncv
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)

                # if transpoes:
                # ax[i,c].plot(mean_fpr, mean_tpr, 'k--', 
                #                lw=2) # label='Mean ROC (area = %0.2f)' % mean_auc,
                # ax[i,c].set_xlim([-0.05, 1.05])
                # ax[i,c].set_ylim([-0.05, 1.05])

                ax[c,i].plot(mean_fpr, mean_tpr, 'k--', 
                               lw=2) # label='Mean ROC (area = %0.2f)' % mean_auc,
                ax[c,i].set_xlim([-0.05, 1.05])  # 
                ax[c,i].set_ylim([-0.05, 1.05])
                # end each plot 

        # set common title and labels
        fig.text(0.5, 0.04, 'False Positive Rate', ha='center')
        fig.text(0.04, 0.5, 'True Positive Rate', va='center', rotation='vertical')
        try: 
            # if title: ax.set_title(title)
            fig.suptitle(title, fontsize=8)
        except: 
            print('ROC.plot2D> missing title.')
        # ax.set_xlabel('False Positive Rate')
        # ax.set_ylabel('True Positive Rate')

        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic (%s)' % identifier)
        # plt.legend(loc="lower right")
        # fig.tight_layout()

        # save 
        if fname is None: fname = ROC.nameROCPlot(**kargs) 
        if kargs.get('save_', True): 
            opath = os.path.join(path, fname)
            print('ROC.plot2D> saving plot to %s' % opath)
            plt.savefig(opath, bbox_inches='tight')  # [2]
        elif show_: 
            plt.show()
        return
    
    @staticmethod
    def getstatsmap(level=0):
        return ROC.stats[level]

    @staticmethod
    def addstats(level, **kargs):
        mean = kargs.get('mean', None)
        ci = kargs.get('ci', None)
        # code, group = kargs.get('code', None), kargs.get('group', None)
        index = ROC.getId(**kargs)
        if level == 2: index = ROC.entry_level2
        if mean is not None: 
            ROC.stats[level]['mean'][index] = mean
        if ci is not None: 
            ROC.stats[level]['ci'][index] = ci

        return 

    @staticmethod
    def collect(e, level=0, **kargs):  # collect (fpr,tpr)s for plotting ROC curves
        n_folds = kargs.get('n_folds', ROC.getNFold(level))
        print('ROC.collect> n_folds: %d' % n_folds)
        code, gn = kargs.pop('code', None), kargs.pop('group', kargs.pop('gn', kargs.pop('model', None)))
        if level == 0: 
            print('ROC.collect> level=0, dim(e): %s' % len(e))
            assert code is not None and gn is not None
            return ROC.addl0(code=code, group=gn, cvset=e, **kargs)
        elif level == 1: 
            print('ROC.collect> level=1, dim(e): %s' % len(e))
            assert code is not None
            return ROC.addl1(code=code, cvset=e, **kargs) 
        elif level == 2: 
            return ROC.addl2(code=code, cvset=e, **kargs)
        else: 
            print('info> Not collecting data for level %s' % level)
            # do nothing for now 
        return

### End ROC 

# plot top 16 and bottom 16 
def plotHeadTail(n_head=16, n_tail=16, level=1, **kargs):
    assert level > 0 
    
    message = kargs.get('message_', 'n/a')
    identifier = kargs.get('identifier', 'roc_head%s_tail%s' % (n_head, n_tail))
    ranked_auc = ROC.rank(level=level, **kargs) # ROC.rankl1() 
    
    div("plotHeadTail> Ranking mean AUC score:\n%s\n" % str(ranked_auc), adaptive=False, symbol='*')
    # head, tail = kargs.get('head', n_head), kargs.get('tail', n_tail)
    head_auc, tail_auc = ranked_auc[:n_head], ranked_auc[-n_tail:]
    
    # configure cv fold
    n_folds = kargs.get('n_folds', None)
    if n_folds is not None: ROC.set_cv(n_folds=n_folds, level=level) # no effect if n_folds is not given (None)

    codes = [c for c, _ in head_auc]
    print('plotHeadTail> plotting top %d codes' % len(codes) )

    # [test]
    print('plotHeadTail-test> showing n_folds (message:%s)' % message); ROC.show_cv()

    # cvset2D = ROC.selectl1(codes=codes, dim=2)  # plot 2D in the order of the given codes
    title = 'ROC (ranked %dst to %dth at level %s)' % (1, n_head, level)

    # plot -> select -> assmeble -> iter
    ROC.plot2D(codes=codes, verfiy_=True, save_=True, 
        identifier=identifier, meta='top%d' % n_head, level=level, title=title)

    codes = [c for c, _ in tail_auc]
    print('plotHeadTail> plotting bottom %d codes' % len(codes))
    # cvset2D = ROC.selectl1(codes=codes, dim=2)  # plot 2D 
    title = 'ROC (ranked %dth to the last at level %s)' % (n_tail, level)
    ROC.plot2D(codes=codes, verfiy_=True, save_=True, 
        identifier=identifier, meta='bottom%d' % n_tail, level=level, title=title) 
    return 

def parallel_hist(**kargs):
    pass

def compareAUC(path, prefix='ci', fsep='_', level=1, **kargs): 
    """
    Compare AUCs in histogram. 
    Wrapper of bar/scatter plot @ t_plotly()

    Memo
    ----
    1. Example input files: 

       a. level 0 
       ci-l0_038.8_blood_bt-100-2.csv 
    
       b. level 1
       ci-l1_053.19_bt-100-2.csv

       c. combined 
       ci-l1_combined_bt-100-2.csv

       d. consolidated 
           level 0: ci-l0_bt-100-3.csv
           level 1: ci-l1_bt-100-3.csv
           level 2: 

    2. move this to learnerUtils.py? 

    3. parameters

       x=['Trial 1', 'Trial 2', 'Trial 3']
       y=[3, 6, 4],

       data = [
          go.Scatter(
        x=[1, 2, 3, 4],
        y=[2, 1, 3, 4],
        error_y=dict(
            type='data',
            symmetric=False,
            array=[0.1, 0.2, 0.1, 0.1],
            arrayminus=[0.2, 0.4, 1, 0.2]
                )
            )
         ]

    """
    def display(sorted_list): 
        for k, v in sorted_list: 
            print "%s: %s" % (k, v)
        return

    import glob
    import dfUtils, learnerUtils, learnerConfig
    import predicate, icd9utils
    # import plotly.plotly as py
    # import plotly.graph_objs as go

    lp = 'l%s' % level
    ext = kargs.get('ext', 'csv')
    identifier = kargs.get('identifier', 'bt')
    ci_header = kargs.get('ci_header', configure.Params.ci_header)
    
    root, infile = path, None
    if os.path.isfile(path): 
        root, infile = os.path.dirname(path), os.path.basename(path)

    if not kargs.has_key('identifier'): kargs['identifier'] = learnerConfig.Domain.identifier
    kargs['sort_'] = True   # sort ~ mean auc
    df = learnerUtils.loadStats(path=path, prefix='ci', level=level, **kargs) 

    div(message='compareAUC> Data loading complete. Data dim: %s' % str(df.shape), symbol='*')

    codemap = kargs.get('codemap', {})  # mapping from code to its canonical names
    codedesc = kargs.get('codedesc', {})  # for text attribute

    codes = list(df['code'].values)  # x
    means = list(df['mean'].values)  # y

    cmean, cmedian = np.median(means), np.mean(means)

    stds = list(df['std'].values)
    dms = list(df['delta_m'].values) # y - delta
    dps = list(df['delta_p'].values) # y + delta
    # print('compareAUC-test> check code type: %s' % type(codes[0]))

    n_head = kargs.get('n_head', kargs.get('head', 16))
    n_tail = kargs.get('n_tail', kargs.get('tail', 16))

    print('compareAUC> Naming the x axis with canonical form of labels ...')

    # only want to focus on specific (diagnostic) codes
    only_these_codes = kargs.get('only_these_codes', []) 
    if len(only_these_codes) == 0: 
        rcodes = list(set(only_these_codes)-set(codes))
        assert len(rcodes) == 0, "compareAUC> some candidate codes do not exist: %s" % rcodes
        codes_ = only_these_codes
    else: 
        codes_ = codes
    # default canonical names
    # codes_repr = ['icd9=%s' % c for c in codes if predicate.isNumber(c)]

    codes_repr = codes_[:]
    for i, c in enumerate(codes_): 
        if icd9utils.isCode(c): # predicate.isNumber(c): 
            codes_repr[i] = '%s' % c  # no need to do 'icd9=%s' % c, use type='category'
        else: 
            codes_repr[i] = c
    # codes_repr = ['%s' % c for c in codes]
    for i, c in enumerate(codes_repr): 
        v = codemap.get(codes_[i], None)
        if v is not None: 
            codes_repr[i] = v

    print('compareAUC> Configuring text for bars in the bar chart')
    text = None
    if codedesc: 
        text = [''] * len(codes_)
        for i, c in enumerate(codes_): 
            if icd9utils.isCode(c):
                text[i] = codedesc.get(c, '')
            
    # alist = [(c, means[i], stds[i], dms[i], dps[i]) for i, c in enumerate(codes_repr)]
    # ranked_auc = sorted(alist, key=lambda e:e[1], reverse=True) 
    # head_codes, tail_codes = ranked_auc[:n_head], ranked_auc[-n_tail:]
    # print('compareAUC-test> head_codes:\n%s\n' % head_codes)
    # print('compareAUC-test> tail_codes:\n%s\n' % tail_codes)

    # codes_repr = [e[0] for e in head_codes] + [e[0] for e in tail_codes]
    # means = [e[1] for e in head_codes] + [e[1] for e in tail_codes]

    # rows = df[ (df['model']=='antibio') & (df['code']=='112.5')]
    # print('debug> rows: %s' % rows)
    # print('debug> dps:\n%s\n' % zip(codes, dps)[:50])

    print('compareAUC> n_codes: %d' % len(codes_))
    print('compareAUC-test> final codes vs means:'); display(zip(codes_repr, means))

    div(message='Histogram rendering starts here ...', symbol='*')

    params = {}
    model= None
    if level == 0: 
        model = kargs.get('model', None)  # antibio, blood, microbio, urine 
        if model is not None: params['model'] = Domain.models[model]

    params['x'] = codes_repr 
    params['y'] = means 
    params['array'] = dps 
    params['arrayminus'] = dms 
    if text is not None: params['text'] = text

    # axis range 
    params['axis_range'] = kargs.get('axis_range', [0.5, 1.0])
    print('info> axis range: %s' % params['axis_range'])

    params['color_marker'] = kargs.get('color_marker', None)
    params['color_err'] = kargs.get('color_err', None)

    # title 
    params['title_x'] = kargs.get('title_x', None)
    # add mean and median estimate 
    params['title_x'] = params['title_x'] + '\ngrand mean: %f, median: %f' % (cmean, cmedian)

    params['title_y'] = kargs.get('title_y', None)
    params['plot_type'] = kargs.get('plot_type', None)

    # prefix, level, identifier, suffix, meta
    meta = kargs.get('meta', None)
    fname = ROC.nameDataFile(prefix='roc-hist', level=level, identifier=identifier, 
        ext='pdf', suffix=model, meta=meta)
    params['opath'] = os.path.join(root, fname)

    print('\ninfo> number of codes: %d' % len(codes_))
    print('info> median (of means): %f, grand mean: %f\n' % (cmean, cmedian))
    
    print('compareAUC> Rendering graphics ...')
    # fig, data = t_plotly_basic(params)
    fig, data = t_plotly(params)

    return fig

# [plotly]
def t_plotly_basic(params):
    
    import plotly.graph_objs as go 

    name = params.get('name', None)
    if name is None: name= 'Diagnostic code vs AUC'

    # go.Bar, go.Scatter, go.Histogram
    trace1 = go.Scatter(
        x=params['x'],
        y=params['y'],
        name=name,
        error_y=dict(
            type='data',
            symmetric=False,
            array=params['array'],
            arrayminus=params['arrayminus'], 
            visible = True
            )
        )

    data = [trace1, ]

    layout = go.Layout(
        barmode='group'
    )
    fig = go.Figure(data=data, layout=layout)

    fpath = params.get('opath', params.get('path', 'roc_basic'))
    base, fname = os.path.dirname(fpath), os.path.basename(fpath)
    assert os.path.exists(base)

    # (@) Send to Plotly and show in notebook
    # py.iplot(fig, filename=fname)
    # (@) Send to broswer 
    plot_url = py.plot(fig, filename=fname)

    py.image.save_as({'data': data}, fpath)
 
    return (fig, data)

def t_plotly(params): 
    """

    Params
    ------
    

    Memo
    ----
    1. color codes:  

       #2296E4, #2F22E4: dark blue

       #E3BA22, #E6842A: orange yellow, orange

       #B0E422, #70E422: green 



    """

    # (*) To communicate with Plotly's server, sign in with credentials file
    import plotly.plotly as py
    # (*) Useful Python/Plotly tools
    import plotly.tools as tls
    import plotly.graph_objs as go
    # (*) Graph objects to piece together plots
    # from plotly.graph_objs import *

    print('plotly> setting color scheme ...')
    color_marker = params.get('color_marker', None)
    # if color_marker is None: color_marker = '#E3BA22'
    color_err = params.get('color_err', None)
    # if color_err is None: color_err = '#E6842A' 

    print('plotly> selecting plot type ...')
    plot_type = params.get('plot_type', 'bar')
    plotFunc = go.Bar
    if plot_type.startswith('b'): 
        # plotFunc = go.Bar
        plot_type = 'bar'
    elif plot_type.startswith('sc'):
        print('plotly> selected scatter plot.')
        plotFunc = go.Scatter 
        plot_type = 'scatter'

    trace_params = params.get('trace_params', None)
    if trace_params is None: 
        trace_params = params

    # Make a Bar trace object
    if color_marker and color_err: 
        # text, opacity
        trace1 = plotFunc(
            x=trace_params['x'],  # a list of string as x-coords
            y=trace_params['y'],   # 1d array of numbers as y-coords
            marker=go.Marker(color=color_marker),  # set bar color (hex color model)
            error_y=go.ErrorY(
                type='data',     # or 'percent', 'sqrt', 'constant'
                symmetric=False,
                array=trace_params.get('array', None),
                arrayminus=trace_params.get('arrayminus', None), 
                color=color_err,  # set error bar color
                thickness=0.6
           )
        )
    else: # default color (blue bars and black error bars)

        trace1 = plotFunc(
            x=trace_params['x'],  # a list of string as x-coords
            y=trace_params['y'],   # 1d array of numbers as y-coords
            error_y=go.ErrorY(
                type='data',     # or 'percent', 'sqrt', 'constant'
                symmetric=False,
                array=trace_params.get('array', None),
                arrayminus=trace_params.get('arrayminus', None), 
            )
        )

    # Make Data object
    data = [trace1, ] # go.Data([trace1])

    titleX = params.get('title_x', None)
    if titleX is None: 
        model = params.get('model', None)
        if model is None: 
            model = 'Combined'
        else: 
            model = model.title()
        titleX = "AUCs of the %s Model" % model.capitalize() # plot's title

    titleY = params.get('title_y', None)
    if titleY is None: 
        titleY = 'Area under an ROC Curve (AUC)'

    axis_range = params.get('axis_range', [0.0, 1.0])

    # Make Layout object
    if plot_type.startswith('b'): # bar plot
        layout = go.Layout(
            title=titleX,       # set plot title
            showlegend=False,  # remove legend

            xaxis = go.XAxis(
                type = 'category',
                showticklabels=True,
                tickangle=45,
                # tickfont=dict(
                #     family='Old Standard TT, serif',
                #     # size=14,
                #     color='black'
                # )
            ),

            yaxis= go.YAxis(
                title=titleY, # y-axis title
                range=axis_range,               # set range
                zeroline=False,                  # remove thick line at y=0
                gridcolor='white'                # set grid color to white
            ),
           paper_bgcolor='rgb(233,233,233)',  # set paper (outside plot) 
           plot_bgcolor='rgb(233,233,233)',   #   and plot color to grey
       )
    else: # automatic range assignment
        layout = go.Layout(
            title=titleX,       # set plot title
            showlegend=False,  # remove legend
            font=dict(size=11), # family='Courier New, monospace', color='#7f7f7f'

            xaxis = dict(
                type = 'category',
                showticklabels=True,
                tickangle=45,
                # tickfont=dict(
                #     family='Old Standard TT, serif',
                #     # size=14,
                #     color='black'
                # )
                titlefont=dict(
                    # family='Courier New, monospace',
                    size=11,
                    # color='#7f7f7f'
                )
            ),

            yaxis= go.YAxis(
                title=titleY, # y-axis title
                range=axis_range,               # set range
                zeroline=False,                  # remove thick line at y=0
                gridcolor='white'                # set grid color to white
            ),

           paper_bgcolor='rgb(233,233,233)',  # set paper (outside plot) 
           plot_bgcolor='rgb(233,233,233)',   #   and plot color to grey
       ) 


    # Make Figure object
    fig = go.Figure(data=data, layout=layout)

    # save file
    fpath = params.get('opath', params.get('path', 'roc_bar'))
    base, fname = os.path.dirname(fpath), os.path.basename(fpath)
    assert os.path.exists(base)

    # (@) Send to Plotly and show in notebook
    # py.iplot(fig, filename=fname)
    # (@) Send to broswer 
    plot_url = py.plot(fig, filename=fname)
    
    py.image.save_as({'data': data}, fpath)

    return (fig, data) 

def t_plotly2(params): # performance evaluation 
    # (*) To communicate with Plotly's server, sign in with credentials file
    import plotly.plotly as py

    # (*) Useful Python/Plotly tools
    import plotly.tools as tls

    import plotly.graph_objs as go
    # (*) Graph objects to piece together plots
    # from plotly.graph_objs import *

    print('plotly> setting color scheme ...')
    color_marker = params.get('color_marker', None)
    # if color_marker is None: color_marker = '#E3BA22'
    color_err = params.get('color_err', None)
    # if color_err is None: color_err = '#E6842A' 

    print('plotly> selecting plot type ...')
    plotFunc = params.get('plot_type', 'bar')
    plot_type = 'bar'
    if plotFunc.startswith('b'): 
        plotFunc = go.Bar
    elif plotFunc.startswith('sc'):
        print('plotly> selected scatter plot.')
        plotFunc = go.Scatter 
        plot_type = 'scatter'

    # Make a Bar trace object
    traces = params.get('traces', None)
    data = []
    for trace_params in traces: 
        color_marker_eff = trace_params.get('color_marker', color_marker)
        if color_marker_eff and color_err: 
            # text, opacity
            trace = plotFunc(
                x=trace_params['x'],  # a list of string as x-coords
                y=trace_params['y'],   # 1d array of numbers as y-coords
                marker=go.Marker(color=color_marker_eff),  # set bar color (hex color model)
                error_y=go.ErrorY(
                    type='data',     # or 'percent', 'sqrt', 'constant'
                    symmetric=False,
                    array=trace_params.get('array', None),
                    arrayminus=trace_params.get('arrayminus', None), 
                    color=color_err,  # set error bar color
                    thickness=0.6
               )
            )
        else: # default color (blue bars and black error bars)
            trace = plotFunc(
                x=trace_params['x'],  # a list of string as x-coords
                y=trace_params['y'],   # 1d array of numbers as y-coords
                marker=go.Marker(color=color_marker_eff), 
                error_y=go.ErrorY(
                    type='data',     # or 'percent', 'sqrt', 'constant'
                    symmetric=False,
                    array=trace_params.get('array', None),
                    arrayminus=trace_params.get('arrayminus', None), 
                )
            )

        # Make Data object
        data.append(trace)
        # data = [trace1, ] # go.Data([trace1])

    titleX = params.get('title_x', None)
    if titleX is None: 
        model = params.get('model', None)
        if model is None: 
            model = 'Combined'
        else: 
            model = model.title()
        titleX = "AUCs of the %s Model" % model # plot's title

    titleY = params.get('title_y', None)
    if titleY is None: 
        titleY = 'Area under the Curve'

    axis_range = params.get('axis_range', None)

    # Make Layout object
    if plot_type.startswith('b'): 
        print('info> configuring layout for bar plot ...')
        layout = go.Layout(
            title=titleX,       # set plot title
            showlegend=False,  # remove legend
            font=dict(size=11), # family='Courier New, monospace', color='#7f7f7f'

            xaxis = go.XAxis(
                type = 'category',
                showticklabels=True,
                tickangle=45,
                # tickfont=dict(
                #     family='Old Standard TT, serif',
                #     # size=14,
                #     color='black'
                # )
            ),

            yaxis= go.YAxis(
                title=titleY, # y-axis title
                range=axis_range,               # set range
                zeroline=False,                  # remove thick line at y=0
                gridcolor='white'                # set grid color to white
            ),
           paper_bgcolor='rgb(233,233,233)',  # set paper (outside plot) 
           plot_bgcolor='rgb(233,233,233)',   #   and plot color to grey
       )
    else: # automatic range assignment
        layout = go.Layout(
            title=titleX,       # set plot title
            showlegend=False,  # remove legend
            font=dict(size=11), # family='Courier New, monospace', color='#7f7f7f'

            xaxis = dict(
                type = 'category',
                showticklabels=True,
                tickangle=45,
                # tickfont=dict(
                #     family='Old Standard TT, serif',
                #     # size=14,
                #     color='black'
                # )
            ),

            yaxis= go.YAxis(
                title=titleY, # y-axis title
                range=axis_range,               # set range
                zeroline=False,                  # remove thick line at y=0
                gridcolor='white'                # set grid color to white
            ),

           paper_bgcolor='rgb(233,233,233)',  # set paper (outside plot) 
           plot_bgcolor='rgb(233,233,233)',   #   and plot color to grey
       ) 


    # Make Figure object
    fig = go.Figure(data=data, layout=layout)

    # save file
    fpath = params.get('opath', params.get('path', 'roc_bar'))
    base, fname = os.path.dirname(fpath), os.path.basename(fpath)
    assert os.path.exists(base)

    # (@) Send to Plotly and show in notebook
    # py.iplot(fig, filename=fname)
    # (@) Send to broswer 
    plot_url = py.plot(fig, filename=fname)
    
    py.image.save_as({'data': data}, fpath)

    return (fig, data) 

    
def set_plt_default(**kargs): 
    # import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = 10, 7.5
    plt.rcParams['axes.grid'] = True
    plt.gray()
    return plt

def plotROC(cvset, clear_plot=True, **kargs): 
    """
    Given a set of (fpr, tpr)s dervied from running cross validation 
    with a classifier, compute the AUC and plot the k-fold CV curves. 
    """
    path = kargs.get('path', DataDir)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    # aucs = []
    # ax = kargs.get('ax', None)
    identifier = kargs.get('identifier', None)

    if clear_plot: plt.clf()

    n_folds_eff = len(cvset)
    # fig, ax = ax.subplots(1, n_groups)   # 1 row of n_group plots
    for i, e in enumerate(cvset):  # foreach cv result where each result has a (fpr, tpr)
        fpr, tpr = e[0], e[1]
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        if n_folds_eff <= 5: 
            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % ( (i+1), roc_auc))
        else: 
            if i % 2 == 0: 
                plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % ( (i+1), roc_auc))
        # aucs.append(roc_auc)

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    mean_tpr /= len(cvset)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (%s)' % identifier)
    plt.legend(loc="lower right")

    # save 
    if fname is None: fname = ROC.nameROCPlot(**kargs) 
    if kargs.get('save_', True): 
        opath = os.path.join(path, fname)
        print('plotROC> saving plot to %s' % opath)
        plt.savefig(opath, bbox_inches='tight')  # [2]
    elif show_: 
        plt.show()

    return mean_auc

def plotROC1D(cvsets, clear_plot=True, **kargs):
    """
    Plot ROCs, which is juxtoposed in a row. 

    Use
    ---
    Compute AUCs and plot ROC curves for a set of group models. 

    Arguments
    ---------
    cvsets: a set of cvset (see plotROC)

    Memo
    ----

    2-fold CV, 3 groups
    [ [(fpr1,tpr1), (fpr2,tpr2)], [(fpr1,tpr1), (fpr2,tpr2)], [(fpr1,tpr1), (fpr2,tpr2)] ]

    """
    path = kargs.get('path', DataDir)
    identifier = kargs.get('identifier', None)

    n_groups = len(cvsets)  # each group model has a cvset
    if clear_plot: plt.clf()
        
    fig, ax = plt.subplots(1, n_groups, sharey=True)   # 1 row of n_group plots

    # for i, a in enumerate(ax.flatten()):
    # scores = []
    for i, cvset in enumerate(cvsets):   # foreach model
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        ncv = len(cvset)
        for j, e in enumerate(cvset): # foreach fold
            fpr, tpr = e[0], e[1]
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            ax[i].plot(fpr, tpr, lw=1)  # label='ROC fold %d (area = %0.2f)' % ( (i+1), roc_auc)

        ax[i].plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        mean_tpr /= ncv
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        ax[i].plot(mean_fpr, mean_tpr, 'k--', 
                    label='Mean ROC (area = %0.2f)' % mean_auc, 
                    lw=2) # label='Mean ROC (area = %0.2f)' % mean_auc,
        ax[i].set_xlim([-0.05, 1.05])
        ax[i].set_ylim([-0.05, 1.05])
        # end each plot in a row

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (%s)' % identifier)
    plt.legend(loc="lower right")

    # save 
    if fname is None: fname = ROC.nameROCPlot(**kargs) 
    if kargs.get('save_', True): 
        opath = os.path.join(path, fname)
        print('ROC.plotROC1D> saving plot to %s' % opath)
        plt.savefig(opath, bbox_inches='tight')  # [2]
    elif show_: 
        plt.show()

    return 

def plotROC2D(cvset2D, clear_plot=True, new_plt=False, **kargs): 
    """
    Plot ROCs in 2D layout. 

    Arguments
    ---------
    adict: a dict with index: (code, group)

    Memo
    ----

    [ [ [(fpr1,tpr1), (fpr2,tpr2)], [(fpr1,tpr1), (fpr2,tpr2)], [(fpr1,tpr1), (fpr2,tpr2)] ] <code1>
      [ [(fpr1,tpr1), (fpr2,tpr2)], [(fpr1,tpr1), (fpr2,tpr2)], [(fpr1,tpr1), (fpr2,tpr2)] ] <code2> 
         ...    
    ]

    """
    path = kargs.get('path', DataDir)
    identifier = kargs.get('identifier', None) 

    n_codes = len(cvset2D)   
    n_groups = len(cvset2D[0])  # each group model has a cvset

    if clear_plot: plt.clf()
        
    fig, ax = plt.subplots(n_codes, n_groups, sharex=True, sharey=True)   

    # for i, a in enumerate(ax.flatten()):
    # scores = []
    for c, cvsets in enumerate(cvset2D): # foreach code 
        for i, cvset in enumerate(cvsets):   # foreach model
            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            ncv = len(cvset)
            for j, e in enumerate(cvset): # foreach fold
                fpr, tpr = e[0], e[1]
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                ax[c,i].plot(fpr, tpr, lw=1)  # label='ROC fold %d (area = %0.2f)' % ( (i+1), roc_auc)

            ax[c,i].plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  # diagnoal
            mean_tpr /= ncv
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            ax[c,i].plot(mean_fpr, mean_tpr, 'k--', 
                        label='Mean ROC (area = %0.2f)' % mean_auc, 
                        lw=2) # label='Mean ROC (area = %0.2f)' % mean_auc,
            ax[c,i].set_xlim([-0.05, 1.05])
            ax[c,i].set_ylim([-0.05, 1.05])
            # end each plot in a row

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (%s)' % identifier)
    plt.legend(loc="lower right")

    # save 
    if fname is None: fname = ROC.nameROCPlot(**kargs) 
    if kargs.get('save_', True): 
        opath = os.path.join(path, fname)
        print('ROC.plotROC2D> saving plot to %s' % opath)
        plt.savefig(opath, bbox_inches='tight')  # [2]
    elif show_: 
        plt.show()
    
    return

def bootstrap_auc_score(y_true, y_pred, n_folds=None, **kargs): 
    """

    Note
    ----
    1. if n_folds = 5, then in each iteration, we only need 
       n_bootstraps/n_folds examples assuming that the size 
       of desired sample is fixed.  
    """
    import math
    # performance metrics 
    # from sklearn.metrics import roc_curve, auc, roc_auc_score

    n_total = kargs.get('n_total', 1200)
    assert len(y_true) == len(y_pred), "y_true:\n%s\ny_pred:\n%s\n" % (y_true, y_pred)
    
    # some parameters
    n_bootstraps_min = 5
    n_labels = 2  # positive and negative label

    if not n_folds: n_folds = ROC.n_folds
    if n_folds < 1: n_folds = 1
    if n_folds > n_total: 
        print('warning> n_folds is large: %d > intended n_sample: %d (in LOO mode?)' % (n_folds, n_total))
    n_bootstraps = int(math.ceil(n_total/(n_folds+0.0))) # [1]
    if n_bootstraps < n_bootstraps_min: n_bootstraps = n_bootstraps_min

    rng_seed = 53  # control reproducibility
    bootstrapped_scores = []

    average = kargs.get('average', 'macro')
    n_unique = len(np.unique(y_true))
    # print('info> n_bootstraps: %d, n_unique_label: %d' % (n_bootstraps, n_unique))
    assert n_unique >= n_labels, "number of unique labels may be too small (incomplete data?): %d" % n_unique

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))  # lowest, highest, n_samples
        n_unique = len(np.unique(y_true[indices]))
        if n_unique < 2: # binary classification
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            # print('warning> at iter=%d number of unique labels is only %d > indices(n=%d):\n%s\n' % \
            #     (i, n_unique, len(indices), indices[:20]))
            continue
    
        # print('info> caller: %s, indices: %s' % (kargs.get('message_'), indices) )
        score = roc_auc_score(y_true[indices], y_pred[indices], average=average)
        bootstrapped_scores.append(score)
        # if i % 100 == 0: print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    # [test]
    n_samples = len(bootstrapped_scores)
    if n_samples < n_bootstraps_min: 
        lc = collections.Counter(y_true)
        for k, v in lc.items(): 
            print('bootstrap-test> label: %s => size: %d' % (k, v))
        raise ValueError, "Number of bootstrap sample is too small: %d" % n_samples

    return bootstrapped_scores

def ci(scores, low=0.05, high=0.95, verbose=False):
    sorted_scores = np.array(scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(low * len(sorted_scores))]
    confidence_upper = sorted_scores[int(high * len(sorted_scores))]
    # if verbose: 
    #     print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    #          confidence_lower, confidence_upper)) 
    return (confidence_lower, confidence_upper)

def ci2(scores, low=0.05, high=0.95, mean=None, verbose=False):
    std = np.std(scores) 
    mean_score = np.mean(scores)
    if mean is None: mean = mean_score

    sorted_scores = np.array(scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(low * len(sorted_scores))]
    confidence_upper = sorted_scores[int(high * len(sorted_scores))]
    middle = (confidence_upper+confidence_lower)/2.0  # assume symmetric

    print('ci2> mean score: %f, middle: %f' % (mean_score, middle))
    # mean = sorted_scores[int(0.5 * len(sorted_scores))]

    # if verbose: 
    #     print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    #         confidence_lower, confidence_upper)) 

    if confidence_upper > 1.0: 
        print('ci2> Warning: upper bound larger than 1.0! %f' % confidence_upper)
        confidence_upper = 1.0

    # this estimate may exceeds 1 
    delminus, delplus = (mean-confidence_lower, confidence_upper-mean)

    return (confidence_lower, confidence_upper, delminus, delplus, std)

def auc_ci(y_true, y_pred, n_folds=5, **kargs): 
    scores = bootstrap_auc_score(y_true=y_true, y_pred=y_pred, n_folds=n_folds, **kargs)
    low, high = kargs.get('low', 0.05), kargs.get('high', 0.95)
    return ci(scores, low=low, high=high)

def collectROCParams(M, R, save_=True, ext='pdf', **kargs): 
    """
    Mainly used to collect (fpr,tpr)s for base model. 
    For higher level models, use stack_ensmble() in the learner module. 
    
    Arguments
    ---------
    M: 3- to 5-tuple: (X, y, classifier, cv, n_folds)
    code, group: 
        index into proper set of (fpr, tpr)s for plotting ROC curve

    Input
    -----
    e.g. M =  (X, y, H[gn], None, n_folds) # X, y, classifier, cv, n_folds
        R =  (code, gn, 0) # code, group, level 

    """
    def v_params(title=None):
        if title is None: title = 'collect> Parameter Profile' 
        div(message=title, symbol='=')
        msg = 'Code: %s, group: %s, level: %s\n' % (code, group, level)
        msg += 'Classifier: %s, cv: %s, n_folds: %d\n' % (classifier, str(cv), n_folds)
        msg += 'Data> dim X: %s y: %d\n' % (str(X.shape), str(y.shape))

        return 

    verbose = kargs.get('verbose', False)

    # model parameters 
    X = y = classifier = cv = n_folds = None
    n_params = len(M) # n_cvparams = len(CV)
    if n_params == 3: 
        X, y, classifier = M
    elif n_params == 4: 
        X, y, classifier, cv = M 
    elif n_params == 5: 
        X, y, classifier, cv, n_folds = M

    assert X is not None and y is not None and classifier is not None 
    if cv is None: 
        if n_folds is None: n_folds = ROC.n_folds
        cv = StratifiedKFold(y, n_folds=n_folds)
    print('collect> n_folds: %d' % n_folds)

    # ROC parameters 
    n_params = len(R)
    code = group = None; level=0
    if n_params == 3: 
        code, group, level = R 
        if group is not None: assert level == 0, "collect> with both code and group, level must be 0 but given level: %d" % level
    elif n_params == 2: 
        code, level = R
        if code is not None: assert level == 1

    ROC.set_cv(n_folds=n_folds, level=level)

    identifier = 'roc'
    if kargs.has_key('identifier'): identifier=kargs['identifier']
    
    if verbose: v_params()

    ### main #### 
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    cvset = []
    n_folds = len(cv)
    auc_scores = []   # for estimating CI
    for i, (train, test) in enumerate(cv):
        # if i == 0: print('info> type(train):%s, train:\n%s\n' % (type(train), train))
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve

        y_true, y_pred = y[test], probas_[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)  

        # estimate CI 
        auc_scores.extend(bootstrap_auc_score(y_true, y_pred, n_folds=n_folds)) 
        if i == 0: print('collect-info> roc_auc: %f ~? auc_scores: %s' % (roc_auc, auc_scores[:5]))

        # mean 
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

        cvset.append( (fpr,tpr) )

    # mean 
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    # bootstrap CI estimate
    ci_auc = ci(auc_scores, low=0.05, high=0.95)

    # add to appropriate map 
    ROC.addstats(level=level, code=code, group=group, ci=ci_auc, mean=mean_auc)
    ROC.add(code=code, group=group, cvset=cvset, level=level, **kargs)
    ROC.verify(level=level, n_folds=n_folds)
    return cvset

def meanROC(cvset): 
    """
    Similar to ROCCV after having TPR and FPR estimates with classifier and training sets. 
    """
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    # aucs = []
    for e in cvset: 
        fpr, tpr = e[0], e[1]
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        # roc_auc = auc(fpr, tpr)
        # aucs.append(roc_auc)
    mean_tpr /= len(cvset)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    return mean_auc

def ROCCV(X, y, classifier=None, cv=None, n_folds=5, 
            show_=False, save_=True, fname=None, ext='pdf', **kargs):
    """

    Arguments
    ---------
    *save_: if True, save the plot

    Note
    ----
    1. Supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    2. bbox_inches='tight': 
       to get rid of redundant, undesirable whitespace around the image. 

    Memo
    ----
    1. StratifiedKFold: 
       <ref> http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedKFold.html  
             Kfold; The folds are made by preserving the percentage of samples for each class. 
    2. ROC: 

    Related
    -------
    ROCCV2
    ROCCVf 

    """ 
    path = kargs.get('path', DataDir)

    if cv is None: 
        cv = StratifiedKFold(y, n_folds=n_folds)
    if classifier is None: 
        classifier = svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state)
    print("ROCCV> input classifier: %s" % classifier)
    try:
        print("ROCCV> params of classifier: %s" % classifier.get_params()) 
    except: 
        pass 
    identifier = 'roc'
    if kargs.has_key('identifier'): identifier=kargs['identifier']

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    # set_plt_default()
    plt.clf()  # clear previous plots
    n_fold_eff = len(cv)

    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        if n_fold_eff <= 5: 
            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % ( (i+1), roc_auc))
        else: 
            if i % 2 == 0: # only plot even folds
                plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % ( (i+1), roc_auc))


    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
              label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (%s)' % identifier)
    plt.legend(loc="lower right")

    # identifier = ('group-%s' % gn) if case is None else (str(case) + '-' + 'group-%s' % gn)
    if fname is None: fname = ROC.nameROCPlot(**kargs) 
    if save_: 
        opath = os.path.join(path, fname)
        print('ROCCV> saving plot to %s' % opath)
        plt.savefig(opath, bbox_inches='tight')  # [2]
    elif show_: 
        plt.show()

    fig = plt.figure()
    plt.close()

    # return fig
    return mean_auc

def t_overlay(): 
    x = [random.gauss(3,1) for _ in range(400)]
    y = [random.gauss(4,2) for _ in range(400)]

    bins = np.linspace(-10, 10, 100)

    pyplot.hist(x, bins, alpha=0.5, label='x')
    pyplot.hist(y, bins, alpha=0.5, label='y')
    pyplot.legend(loc='upper right')
    pyplot.show()

def save_plot(plt, fname=None, **kargs):
    path = os.path.join(os.getcwd(), 'plot')
    if not os.path.exists(path): os.makedirs(path)

    if not fname: fname = 'test.pdf'
    plt.savefig(os.path.join(path, fname), bbox_inches='tight')
    return plt

def t_subplot(**kargs): 
    from matplotlib import pyplot as plt

    fig = plt.figure()

    ax1 = fig.add_subplot(221)   # middle digist: 1 row 2 column
    ax1.plot([(1, 2), (3, 4)], [(4, 3), (2, 3)])
    ax2 = fig.add_subplot(222)
    ax2.plot([(7, 2), (5, 3)], [(1, 6), (9, 5)])

    if kargs.get('save', True): 
        plt = save_plot(plt, fname='subplot_test.pdf')
    else: 
        plt.show()

    return 

def demo_subplots3(**kargs): 
    fig, axes2d = plt.subplots(nrows=3, ncols=3,
                           sharex=True, sharey=True,
                           figsize=(6,6))

    for i, row in enumerate(axes2d):
        for j, cell in enumerate(row):
            cell.imshow(np.random.rand(32,32))

    fig.text(0.5, 0.04, 'common X', ha='center')
    fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')

    plt.tight_layout()
    return

def demo_subplots2(**kargs): 
    import matplotlib.pyplot as plt

    x = np.arange(0,10.0, 0.1)
    fig, ax = plt.subplots(2,2)   # ax is a numpy array
    ax[0,0].plot(x, x)
    ax[0,1].plot(x, x**2)
    ax[1,0].plot(x, np.sqrt(x))
    ax[1,1].plot(x, 1./(x+0.01))
    # plt.show()

    path = os.path.join(os.getcwd(), 'plot')
    if not os.path.exists(path): os.makedirs(path)

    if kargs.get('save', True): 
        plt = save_plot(plt, fname='subplot_2by2.pdf')
        # doesn't save if plt.show() called before this
    else: 
        plt.show()

    return

def demo_subplots(**kargs): 
    """Examples illustrating the use of plt.subplots().

    This function creates a figure and a grid of subplots with a single call, while
    providing reasonable control over how the individual plots are created.  For
    very refined tuning of subplot creation, you can still use add_subplot()
    directly on a new figure.
    """
    import matplotlib.pyplot as plt

    # Simple data to display in various forms
    x = np.linspace(0, 2 * np.pi, 400)
    y = np.sin(x ** 2)

    plt.close('all')

    # Just a figure and one subplot
    f, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Simple plot')

    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(x, y)
    axarr[0].set_title('Sharing X axis')
    axarr[1].scatter(x, y)

    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(x, y)
    ax1.set_title('Sharing Y axis')
    ax2.scatter(x, y)

    # Three subplots sharing both x/y axes
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    ax1.plot(x, y)
    ax1.set_title('Sharing both axes')
    ax2.scatter(x, y)
    ax3.scatter(x, 2 * y ** 2 - 1, color='r')

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    # row and column sharing
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax1.plot(x, y)
    ax1.set_title('Sharing x per column, y per row')
    ax2.scatter(x, y)
    ax3.scatter(x, 2 * y ** 2 - 1, color='r')
    ax4.plot(x, 2 * y ** 2 - 1, color='r')

    # Four axes, returned as a 2-d array
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(x, y)
    axarr[0, 0].set_title('Axis [0,0]')
    axarr[0, 1].scatter(x, y)
    axarr[0, 1].set_title('Axis [0,1]')
    axarr[1, 0].plot(x, y ** 2)
    axarr[1, 0].set_title('Axis [1,0]')
    axarr[1, 1].scatter(x, y ** 2)
    axarr[1, 1].set_title('Axis [1,1]')
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    # Four polar axes
    plt.subplots(2, 2, subplot_kw=dict(polar=True))

    plt.show()

def t_axis(): 
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    pos=1000
    c=0.1
    # trap1=trapping(pos,c)  #instance of class trapping
    # plot1d=trap1.steps1d(pos,c) #use the method steps1d from class
    plot1d=np.linspace(0,25,100)

    mylabel=('c=0.1','c=0.01','c=0.001') 
    colors=('bo','ro','mo')
    cn=stats.norm.sf(plot1d)  #create the survival function
    for label,color in zip(mylabel,colors):
        # plt.loglog(plot1d,cn,color,label=label)
        plt.semilogx(plot1d,cn,color,label=label)

    plt.show()

def t_roc(): 
    import predicate
    ROC.n_folds = 3
    cvset = [(0.1, 0.3), (0.2, 0.4), (0.1, 0.3)]

    ROC.addl1(code='00.0', cvset=cvset)
    ROC.verify(level=1)

def t_cluster_demo(): 
    import scipy
    import pylab
    import scipy.cluster.hierarchy as sch

    # Generate random features and distance matrix.
    x = scipy.rand(40)
    D = scipy.zeros([40,40])
    for i in range(40):
        for j in range(40):
            D[i,j] = abs(x[i] - x[j])

    # Compute and plot first dendrogram.
    fig = pylab.figure(figsize=(8,8))
    ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
    Y = sch.linkage(D, method='centroid')
    Z1 = sch.dendrogram(Y, orientation='right')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
    Y = sch.linkage(D, method='single')
    Z2 = sch.dendrogram(Y)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    D = D[idx1,:]
    D = D[:,idx2]
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    pylab.colorbar(im, cax=axcolor)
    fig.show()
    fig.savefig('dendrogram.pdf')

    return



def t_histogram(**kargs): 
    
    # source: learnerTestMod.t_metric_iter()
    ifile = 'performance_test.pkl'
    rootdir = '.'
    fpath = os.path.join(rootdir, ifile)
    assert os.path.exists(fpath)
    adict = pickle.load(open(fpath, "rb" ))
    # adict[model][code] = {'median': med, 'mean': avg}
    assert len(adict) > 0

    x0, x1 = [], []
    xy0, xy1 = [], []
    acc = 0
    for model, entry in adict.items(): 
        for code, metric in entry.items(): 
            x0.append(metric['median'])
            x1.append(metric['mean'])
            xy0.append((code, metric['median']))
            xy1.append((code, metric['mean']))

    print('info> medians: %s' % x0)
    print('info> means:   %s' % x1)
    overlaid_hist(x0=x0, x1=x1, xy0=xy0, xy1=xy1)

    return

def overlaid_hist(**kargs): 
    import plotly.plotly as py
    import plotly.graph_objs as go

    np.random.seed(53)

    x0 = kargs.get('x0', np.random.randn(500))
    x1 = kargs.get('x1', np.random.randn(500)+1)
    fname = kargs.get('filename', 'overlaid-histogram')

    trace1 = go.Histogram(
        x=x0,
        opacity=0.75
    )
    
    trace2 = go.Histogram(
        x=x1,
        opacity=0.75
    )
    data = [trace1, trace2]
    layout = go.Layout(
        barmode='overlay'
    )
    fig = go.Figure(data=data, layout=layout)
    plot_url = py.plot(fig, filename=fname)

    return

def t_histogram2(**kargs): 
    # import pylab as P

    # source: learnerTestMod.t_metric_iter()
    rootdir = '.'
    ifiles = ['performance_test-ms.pkl', ]  # 'performance_test-cross.pkl'
    params = {}
    params['traces'] = []
    for ifile in ifiles: 
        # ifile = 'performance_test-%s.pkl' % mode

        fpath = os.path.join(rootdir, ifile)
        assert os.path.exists(fpath)
        adict = pickle.load(open(fpath, "rb" ))
        # adict[model][code] = {'median': med, 'mean': avg}
        assert len(adict) > 0

        y0, y1 = [], []
        xy0, xy1 = [], []
        x = []
        acc = 0
        for model, entry in adict.items(): 
            for code, metric in entry.items(): 
                x.append(code)
                y0.append(metric['median'])
                y1.append(metric['mean'])
                xy0.append((code, metric['median']))
                xy1.append((code, metric['mean']))

        print('info> medians: %s' % y0)
        print('info> means:   %s' % y1)

        trace = {}
        trace['x'] = x 
        trace['y'] = y0

        params['traces'].append(trace)

    params['model'] = 'microbio'

    # trace1 = {}
    # trace1['x'] = x 
    # trace1['y'] = y0
    # trace2 = {}
    # trace2['x'] = x
    # trace2['y'] = y1
    # params['traces'] = [trace1, trace2]  

    # if text is not None: params['text'] = text

    params['color_marker'] = kargs.get('color_marker', None)
    params['color_err'] = kargs.get('color_err', None)

    # title 
    params['title_x'] = kargs.get('title_x', 'Test Codes (Self vs Cross)')
    params['title_y'] = kargs.get('title_y', 'AUCs')
    params['plot_type'] = kargs.get('plot_type', 'bar')

    # prefix, level, identifier, suffix, meta
    meta = kargs.get('meta', None)
    fname = '%s-performance-hist.pdf' % params['model']
    params['opath'] = os.path.join(rootdir, fname)

    t_plotly2(params)
    # y = np.vstack((y0, y1))
    # y = y.T
    # P.figure()
    # n, bins, patches = P.hist(x, y, normed=1, histtype='bar',
    #                         color=['burlywood', 'Chartreuse'],
    #                         label=['Median', 'Mean'])
    # P.legend(loc='best') 
    # P.show()
    # # P.savefig(os.path.join('.', 'performance_test.png'), bbox_inches='tight') 

    return 

def t_data_profile(**kargs): 
    """
        1. bar plot 
       #F2F0CE   : yellow urine 
       #C8E3EF   : blue antibio
       #D0EFDA   : green microbio
       #EED0D7   : red blood

       #BFC3E3   : light blue/violet, combined at level 1

    2. scatter map 
       #5E88DB   : blue curve, antibio

       #283BE5   : dark blue, combined at level 1

    Memo
    ----
    1. Color picker: http://www.w3schools.com/colors/colors_picker.asp

    """
    import pandas as pd

    rootdir = '.'

    ifiles = []
    active_models = ['microbio', 'antibio', 'blood', 'urine',  ]  
    active_models_std= ['Microbiology', 'Antibiotic', 'Blood Test', 'Urine Test', ]
    sortbyattr = 'nuniq' # number of unique patients
    metrics = ['nuniq', 'nrow', ]

    color_markers = {'microbio': ['#00b300', '#00cc99'], 
                     'antibio': ['#0066ff', '#66ccff'], 
                     'urine': ['#ff9900', '#ffcc80'],     
                     'blood': ['#ff0066', '#ff99cc']}  
                     # ['#C8E3EF', '#F2F0CE']
    f_dtype = {'code': str}

    # 'tset_stats_antibio-sort_by_nuniq.csv'
    for m in active_models: 
        ifile = 'tset_stats_%s-sort_by_%s.csv' % (m, sortbyattr)
        fpath = os.path.join(rootdir, ifile)
        assert os.path.exists(fpath), 'invalid path: %s' % fpath
        ifiles.append(fpath)  # 'performance_test-cross.pkl'

    nrows_total = nuniq_total = 0
    for i, fpath in enumerate(ifiles): 
        # ifile = 'performance_test-%s.pkl' % mode

        params = {}
        params['traces'] = []

        assert os.path.exists(fpath)
        
        df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True, dtype=f_dtype)
        df.sort_values(sortbyattr, ascending=True, inplace=True) 

        inrow = df['nrow'].sum()
        nrows_total += inrow
        print('info> model: %s, nrow: %d' % (active_models[i], inrow))
        nuniq_total += df['nuniq'].sum()

        x = df['code'].values
        # nuniq = df['nuniq'].values 
        # nrow = df['nrow'].values

        print('info> code (size: %d):\n%s' % (len(x), x))

        # trace_params = {}
        for j, metric in enumerate(metrics): 
            trace = {}
            trace['x'] = x 
            trace['y'] = df[metric].values 

            trace['color_marker'] = color_markers[active_models[i]][j]

            print('info> metric=%s:\n%s' % (metric, trace['y']))
            params['traces'].append(trace)
            
            # # [test]
            # trace_params['x'] = x 
            # trace_params['y'] = df[metric].values 
            # trace_params['color_marker'] = color_markers[active_models[i]][j]

        params['model'] = active_models[i]

        # if text is not None: params['text'] = text

        params['color_marker'] = kargs.get('color_marker', None)
        params['color_err'] = kargs.get('color_err', None)

        # title 
        params['title_x'] = kargs.get('title_x', 'Training Set Profile for the %s Model' % active_models_std[i])
        params['title_y'] = kargs.get('title_y', 'Number of Unique Patients vs Training Set Size')
        params['plot_type'] = kargs.get('plot_type', 'bar')

        # prefix, level, identifier, suffix, meta
        meta = kargs.get('meta', None)
        fname = '%s-data-hist.pdf' % params['model']
        params['opath'] = os.path.join(rootdir, fname)


        params['axis_range'] = [10, 20000] # axis_range
        # params['plot_type'] = 'scatter'

        # params['trace_params'] = trace_params
        
        # t_plotly(params)
        t_plotly2(params)

    print('info> total nrow: %d, total n_patients: %d' % (nrows_total, nuniq_total))

    return

def t_roc_multiclass(): 
    """

    Memo
    ----
    error bar 
      #0F100F: dark gray

    1. scatter map 
    #8E51F0: violet

    2. bar plot
       <very light marker> 
    
       #E2D7F3: violet
    """
    path = '/Users/pleiades/Documents/work/cumc/bulk_training-analysis/roc' 
   
    level, n_trial = (1, 3)

    # ftb = {}   # file generated by: learnerUtils.consolidate()? 
    # ftb[0] = 'ci-l0_bt-100-%s.csv' % n_trial
    # ftb[1] = 'ci-l1_bt-100-%s.csv' % n_trial
    fname = 'ci-l1_multiclass-1vs1_bt-100.csv'

    title_x = None # default, AUCs of the %s Factor
    if level > 0: 
        title_x = 'AUCs of the Multiclass Model at Level %s' % level

    fp = os.path.join(path, fname)
    # codemap = t_map()
    codemap = {}  # mapping code to canonical form for more interpretable x-axis 

    # text data? 
    # reference: data-feature/icd9-desc-sorted.csv
    # code_desc = mapICD() 

    compareAUC(path=fp, prefix='ci', fsep='_', level=level, codemap=codemap, 
           color_marker='#8E51F0', color_err='#000000', title_x=title_x, plot_type='scatter', codedesc=mapICD())
    

    return 

def t_roc_compare(**kargs):
    """

    



    Memo
    ----
    1. color code 

       #2296E4, #2F22E4: dark blue

       #E3BA22, #E6842A: orange yellow, orange

       #C4DF7C, #67D4A3: green
       #A3E997                   (v)


       #77D3A9
       #72D9AB, #7D9CDF: green + blue
                #A5BCEF
                #B6CAF6

       #27C57E, #57A3DA: green blue v2
                #77B7E5
                #1C6FC8

       #EA7F9A, #F7517A: red
       
       more contrast 
       #F48DA7, #F83566: red
       #F494AC   ..    : red (v)


       #5E88DB, : blue grey
       #4187F0, #4F5052

       <lighter>
       #90E5BF, #6D7C75 : green grey
                #282B2A
                #0F100F

       <very light> 
       error: #0F100F  : dark gray
       marker: #DCF6DF  : green

      # Scattet map
         * level 0 

         #1D935C, #0F100F: green curve

         #E42252: red curve
         #E92B57
         #F03D66

         #F0A53D, #0F100F: orange curve

         * level 1 

         #3654CE 

    # 11.30.15 

    1. bar plot 
       #F2F0CE   : yellow urine 
       #C8E3EF   : blue antibio
       #D0EFDA   : green microbio
       #EED0D7   : red blood

       #BFC3E3   : light blue/violet, combined at level 1

    2. scatter map 
       #5E88DB   : blue curve, antibio

       #283BE5   : dark blue, combined at level 1


    # 2.18.16
    /learner-bt6-t1.6.log:15553:consolidate> saving consolidated file to 
        /phi/proj/poc7002/bulk_training/experiment/data-learner/t1/ci-l0_190-1.csv

    """
    import random 
    random.seed(10)

    mmap = {}
    mmap['microbio'] = 'Microbiology'
    mmap['antibio'] = 'Antibiotic'
    mmap['blood'] = 'Blood Test'
    mmap['urine'] = 'Urine Test'

    n_trial = kargs.get('n_trial', 53)
    # path = '/Users/pleiades/Documents/work/cumc/bulk_training-plot/performance/t%s' % n_trial
    path = '/Users/pleiades/Documents/work/cumc/bulk_training-result/t%s' % n_trial
    # fname = 'ci-l1_bt-100-3.csv'
    level, model = (kargs.get('level', 0), kargs.get('model', None))

    if level == 0: 
        if model is None: model = 'microbio'
        assert model in ('microbio', 'antibio', 'blood', 'urine')
    if level == 1: 
        if model is None or model == 'combined': model = 'global' # or local
        assert model in ('global', 'local')
    if level == 2: 
        model = 'combined'

    # color map 
    color_map = {}
    color_map['blood'] = '#F03D66'
    color_map['antibio'] = '#5E88DB'
    color_map['microbio'] = '#1D935C' # '#A3E997'
    color_map['urine'] = '#E6842A'

    color_map['combined'] = '#283BE5'   # level 1, 2
    color_map['global'] = '#283BE5'  # level 1
    color_map['local'] =  '#8258FA' # violet   # level 1 

    ftb = {}

    # correct 
    # ftb[0] = 'ci-l0_bt_190-%s.csv' % n_trial
    # ftb[1] = 'ci-l1_bt_190-%s.csv' % n_trial

    # temp 
    ftb[0] = 'ci-l0_bt_190.csv' 
    if level == 1: 
        ftb[1] = 'ci-l1_bt_190.csv' if model in ('global', ) else 'ci-l1_bt_190-local.csv'
    # ftb[2] = 'ci-l2_bt_190.csv'

    only_these_codes = kargs.get('only_these_codes', [])
    title_x = kargs.get('title_x', None) # default, AUCs of the %s Factor
    if title_x is None: 
        if level == 0: 
            title_x = 'Sorted Performance of the %s Model' % mmap.get(model, model.capitalize())
        elif level == 1: 
            assert model in ('global', 'local')
            title_x = 'Sorted Performance of the %s Level-1 Model' % model.capitalize()
    
    fp = os.path.join(path, ftb[level])

    # codemap = t_map()
    codemap = {}  # mapping code to canonical form for more interpretable x-axis 

    # text data? 
    # reference: data-feature/icd9-desc-sorted.csv
    # code_desc = mapICD() 
    compareAUC(path=fp, prefix='ci', fsep='_', level=level, model=model, codemap=codemap, 
               only_these_codes=only_these_codes, 
                color_marker=color_map[model], color_err='#000000', title_x=title_x, plot_type='scatter',
                    axis_range=[0.5, 1.0], 
                    codedesc=mapICD())
    
    # level 1 config 
    # compareAUC(path=fp, prefix='ci', fsep='_', level=1, model='combined', codemap=codemap, 
    #        color_marker=None, color_err=None, title_x=title_x, plot_type='scatter')

    # multiclass at level1 


    return

def t_roc_compare2(**kargs):
    n_trial = 53

    active_models = ['microbio', 'antibio', 'blood', 'urine',  ]
    active_models_std= ['Microbiology', 'Antibiotic', 'Blood Test', 'Urine Test', ]

    cases = [(1, 'local'), ] # (0, 'microbio'), (0, 'antibio'), (0, 'blood'), (0, 'urine'), (1, 'global')
    
    # ldict, udict = t_rank_rarity(**kargs)
    for l, m in cases: 
            # find x rarest and y most common conditions
        t_roc_compare(level=l, model=m)  

    return

def t_roc_compare3(**kargs):

    level, model = kargs.get('level', 1), kargs.get('model', 'local')
    lx, ux = t_rank_rarity(**kargs)
    n_trial = 53

    active_models = ['microbio', 'antibio', 'blood', 'urine',  ]
    active_models_std= ['Microbiology', 'Antibiotic', 'Blood Test', 'Urine Test', ]

    cases = [(level, model), ] # (0, 'microbio'), (0, 'antibio'), (0, 'blood'), (0, 'urine'), (1, 'global'), (1, 'local')
    
    # ldict, udict = t_rank_rarity(**kargs)

    for l, m in cases: 
        # find x rarest and y most common conditions 
        t_roc_compare(level=l, model=m, only_these_codes=lx, title_x='Lowest %d Performance Scores in AUCs among Local Level-1 Models' % len(lx))  

    for l, m in cases: 
        t_roc_compare(level=l, model=m, only_these_codes=ux, title_x='Highest %d Performance Scores in AUCs among Local Level-1 Models' % len(ux)) 

    return

def t_roc_selective_compare(**kargs):
    """
    Same as t_roc_compare() but shows only upper and lower N diagnostic codes. 

    """
    return 

def t_rank_rarity(**kargs):
    """
    
    Returns
    -------
    (lx, ux)
    where lx: lower N codes according to performance score (e.g. mean AUC)
          ux: 

    """
    def display_dict(adict):
        for k, v in adict.items(): 
            print('[%s] %s' % (k, v))

    import pandas as pd
    import random 
    random.seed(10)

    mmap = {}
    mmap['microbio'] = 'Microbiology'
    mmap['antibio'] = 'Antibiotic'
    mmap['blood'] = 'Blood Test'
    mmap['urine'] = 'Urine Test'

    n_trial = kargs.get('n_trial', 53)
    # path = '/Users/pleiades/Documents/work/cumc/bulk_training-plot/performance/t%s' % n_trial
    path = '/Users/pleiades/Documents/work/cumc/bulk_training-result/t%s' % n_trial
    # fname = 'ci-l1_bt-100-3.csv'
    level, model = (kargs.get('level', 0), kargs.get('model', None))
    if level == 0: 
        if model is None: model = 'microbio'
        assert model in ('microbio', 'antibio', 'blood', 'urine')
    if level == 1: 
        if model is None or model == 'combined': model = 'global' # or local
        assert model in ('global', 'local')
    if level == 2: 
        model = 'combined'

    print('t_rank_rarity> level: %s, model: %s' % (level, model))
    # color map 
    color_map = {}
    color_map['blood'] = '#F03D66'
    color_map['antibio'] = '#5E88DB'
    color_map['microbio'] = '#1D935C' # '#A3E997'
    color_map['urine'] = '#E6842A'

    color_map['combined'] = '#283BE5'   # level 1, 2
    color_map['global'] = '#283BE5'  # level 1
    color_map['local'] =  '#8258FA' # violet   # level 1 

    ftb = {}

    # correct 
    # ftb[0] = 'ci-l0_bt_190-%s.csv' % n_trial
    # ftb[1] = 'ci-l1_bt_190-%s.csv' % n_trial

    # temp 
    ftb[0] = 'ci-l0_bt_190.csv' 
    if level == 1: 
        ftb[1] = 'ci-l1_bt_190.csv' if model in ('global', ) else 'ci-l1_bt_190-local.csv'
    # ftb[2] = 'ci-l2_bt_190.csv'

    only_these_codes = kargs.get('only_these_codes', [])
    title_x = kargs.get('title_x', None) # default, AUCs of the %s Factor
    if title_x is None: 
        if level == 0: 
            title_x = 'Sorted Performance of the %s Model' % mmap.get(model, model.capitalize())
        elif level == 1: 
            assert model in ('global', 'local')
            title_x = 'Sorted Performance of the %s Level-1 Model' % model.capitalize()
    
    fp = os.path.join(path, ftb[level])
    
    # sort according to mean AUC  
    sortbyattr = 'mean'
    f_dtype = {'code': str}
    df = pd.read_csv(fp, sep='|', header=0, index_col=False, error_bad_lines=True, dtype=f_dtype)
    df.sort_values(sortbyattr, ascending=True, inplace=True) 

    lowerN, upperN = kargs.get('lower', 10), kargs.get('upper', 10)
    sorted_codes = df['code'].values
    scores = df['mean'].values
    amean, amedian = np.mean(scores), np.median(scores)
    print('> overall mean: %f, median: %f' % (amean, amedian))

    lx = sorted_codes[:lowerN]
    lc = scores[:lowerN]
    lmean, lmedian = np.mean(lc), np.median(lc)
    print('> lowest %d scores (mean: %f, med: %f) => \n%s\n' % (lowerN, lmean, lmedian, zip(lx, lc)))
    ux = sorted_codes[-upperN:]
    uc = scores[-upperN:]
    hmean, hmedian = np.mean(uc), np.median(uc)
    print('> highest %d scores (mean: %f, med: %f) => \n%s\n' % (upperN, hmean, hmedian, zip(ux, uc)))

    return (lx, ux)


def t_rank_compare(**kargs):
    import pandas as pd
    import scipy.stats as stats

    n_trial = kargs.get('n_trial', 53)
    # path = '/Users/pleiades/Documents/work/cumc/bulk_training-plot/performance/t%s' % n_trial
    path = '/Users/pleiades/Documents/work/cumc/bulk_training-result/t%s' % n_trial

    f1 = os.path.join(path, 'ci-l1_bt_190.csv') 
    f2 = os.path.join(path, 'ci-l1_bt_190-local.csv')

    df1 = pd.read_csv(f1, sep='|', header=0, index_col=False, error_bad_lines=True)
    df2 = pd.read_csv(f2, sep='|', header=0, index_col=False, error_bad_lines=True)

    # [test]
    sortbyattr = 'mean'
    df1t = df1.sort_values(sortbyattr, ascending=True, inplace=False)
    df2t = df2.sort_values(sortbyattr, ascending=True, inplace=False)
    print('df1t:\n%s\n' % df1t.head(5))
    print('df2t:\n%s\n' % df2t.head(5))

    df1.sort_values(sortbyattr, ascending=True, inplace=True)
    df2.sort_values(sortbyattr, ascending=True, inplace=True)

    cx1 = df1['code'].values
    cx2 = df2['code'].values

    tau, p_value = stats.kendalltau(cx1, cx2)
    print('info> tau: %f, p-val: %f' % (tau, p_value))


def mapICD(**kargs): 
    """

    Memo
    ----
    File icd9-desc-sorted.csv is located under symlink: data-feature
    """
    import dfUtils, utils

    root = os.getcwd()
    adict = {}
    try: 
        df = dfUtils.load_df(root=root, _file='icd9-desc-sorted.csv', sep='|', verbose=False, dtypes=str)
        codes = df['code'].values
        desc = df['desc'].values
        adict = dict(zip(codes, desc))
        partial = utils.sample_dict(adict, n_sample=10)

        print('info> Sample entries of ICD9 code description/text: ')
        print partial 
    except Exception, e: 
        print('warning> missing map file? error=%s' % e)

    return adict

def t_map(**kargs): 
    adict = {}
    adict['027.0'] = 'Listeriosis (027.0)'
    return adict

def main(): 
    # t_axis()

    # t_subplot()
    # demo_subplots2()

    # comparison of performance at different level of abstraction
    # t_roc()
    # t_roc_compare(level=1, model='global')
    # t_roc_compare2()
    t_roc_compare3(level=1, model='global')

    # t_rank_compare()

    # t_histogram()  # e.g. learner test
    # t_histogram2()

    # multiclass 
    # t_roc_multiclass()

    # mapICD()

    # overlaid histograms 
    # overlaid_hist()


    # plot data distribution (profile)
    # t_data_profile()

    return 

if __name__ == "__main__":
    main()

