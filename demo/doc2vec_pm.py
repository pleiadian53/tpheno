# encoding=utf8  

# import sys  

# reload(sys)  
# sys.setdefaultencoding('utf8')

import locale
import glob
import os.path
import requests
import tarfile
import io, os

import gensim
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple

import unicodedata, re

from doc2vec_utils import *

# all_chars = (unichr(i) for i in xrange(0x110000))
# control_chars = ''.join(c for c in all_chars if unicodedata.category(c) == 'Cc')    

# or equivalently and much more efficiently
control_chars = ''.join(map(unichr, range(0,32) + range(127,160)))
control_char_re = re.compile('[%s]' % re.escape(control_chars))

from config import seq_maker_config, sys_config

################################################################################################################################
#  Demo Paragraph Vector using gensim's Doc2Vec 
# 
#  reference: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb
#
################################################################################################################################

dirname = 'aclImdb'
filename = 'aclImdb_v1.tar.gz'
locale.setlocale(locale.LC_ALL, 'C')


# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text

def remove_ctrl_char(text, repl=''):  # repl: replacement character
    # [global] control_char_re
    return control_char_re.sub(repl, text)

def load(**kargs): 
    if not os.path.isfile('aclImdb/alldata-id.txt'):
        if not os.path.isdir(dirname):
            if not os.path.isfile(filename):
                # Download IMDB archive
                url = 'http://ai.stanford.edu/~amaas/data/sentiment/' + filename
                r = requests.get(url)

                with open(filename, 'wb') as f:
                    f.write(r.content)

            tar = tarfile.open(filename, mode='r')
            tar.extractall()
            tar.close()

        # Concat and normalize test/train data
        folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg', 'train/unsup']
        alldata = u''

        for fol in folders:
            temp = u''
            output = fol.replace('/', '-') + '.txt'

            # Is there a better pattern to use?
            txt_files = glob.glob('/'.join([dirname, fol, '*.txt']))

            for txt in txt_files:
                with io.open(txt, 'r', encoding='utf-8') as t:
                    control_chars = [chr(0x85)]
                    t_clean = t.read()

                    # encode('utf-8').strip()
                    # for c in control_chars:
                    # 	# c = c.encode('utf-8').strip()
                    #     t_clean = t_clean.replace(c, ' ')
                    t_clean = remove_ctrl_char(t_clean, repl=' ')

                    temp += t_clean

                temp += "\n"

            temp_norm = normalize_text(temp)
            with io.open('/'.join([dirname, output]), 'w', encoding='utf-8') as n:
                n.write(temp_norm)

            alldata += temp_norm

        with io.open('/'.join([dirname, 'alldata-id.txt']), 'w', encoding='utf-8') as f:
            for idx, line in enumerate(alldata.splitlines()):
                num_line = "_*{0} {1}\n".format(idx, line)
                f.write(num_line)

    return

def verify_load(**kargs): 
    basedir = 'aclImdb'
    assert os.path.isfile("aclImdb/alldata-id.txt"), "alldata-id.txt unavailable"

def tag_doc(**kargs):
    # import gensim
    # from gensim.models.doc2vec import TaggedDocument
    # from collections import namedtuple

    SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

    alldocs = []  # will hold all docs in original order
    with io.open('aclImdb/alldata-id.txt', encoding='utf-8') as alldata:
        for line_no, line in enumerate(alldata):
            tokens = gensim.utils.to_unicode(line).split()
            words = tokens[1:]
            tags = [line_no] # `tags = [tokens[0]]` would also work at extra memory cost
            split = ['train','test','extra','extra'][line_no//25000]  # 25k train, 25k test, 25k extra
            sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
            alldocs.append(SentimentDocument(words, tags, split, sentiment))

    docs_parts = {}

    train_docs = [doc for doc in alldocs if doc.split == 'train']; docs_parts['train'] = train_docs
    test_docs = [doc for doc in alldocs if doc.split == 'test']; docs_parts['test'] = test_docs
    doc_list = alldocs[:]  # for reshuffling per pass

    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(doc_list), len(train_docs), len(test_docs))) 

    return (alldocs, docs_parts)

def config_model(**kargs): 
    """


    Note
    ----
    
    Parameter choices: 

    1. 100-dimensional vectors, as the 400d vectors of the paper don't seem to offer much benefit on this task; 
       similarly, frequent word subsampling seems to decrease sentiment-prediction accuracy, so it's left out

    2. cbow=0 means skip-gram which is equivalent to the paper's 'PV-DBOW' mode, 
       matched in gensim with dm=0; 
       added to that DBOW model are two DM models, one which averages context vectors (dm_mean) and 
       one which concatenates them (dm_concat, resulting in a much larger, slower, more data-hungry model)

    3. a min_count=2 saves quite a bit of model memory, discarding only words that appear in a single doc 
      (and are thus no more expressive than the unique-to-each doc vectors themselves)

    """
    from gensim.models import Doc2Vec
    import gensim.models.doc2vec
    from collections import OrderedDict
    import multiprocessing
    from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

    alldocs = kargs.get('sequences', kargs.get('alldocs', tag_doc(**kargs)[0]))

    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

    simple_models = [
        # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
        Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DBOW 
        Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DM w/average
        Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
    ]

    # speed setup by sharing results of 1st model's vocabulary scan
    simple_models[0].build_vocab(alldocs)  # PV-DM/concat requires one special NULL word so it serves as template
    print(simple_models[0])
    for model in simple_models[1:]:
        model.reset_from(simple_models[0])
        print(model)

    models_by_name = OrderedDict((str(model), model) for model in simple_models)

    # concatenate models
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]]) # PV-DBOW + PV-DM w/average
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]]) # PV-DBOW + PV-DM w/concatenation

    return models_by_name

def concat_models(models_by_name): 
    from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
    
    simple_models = [
        # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
        Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DBOW 
        Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DM w/average
        Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
    ]    

    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]]) # PV-DBOW + PV-DM w/average
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]]) # PV-DBOW + PV-DM w/concatenation

    return models_by_name

def test(**kargs): 

    load(**kargs)
    verify_load(**kargs)
    alldocs, docs_parts = tag_doc(**kargs)

    models_by_name = config_model(alldocs=alldocs) 

    return 

if __name__ == "__main__":
    test()