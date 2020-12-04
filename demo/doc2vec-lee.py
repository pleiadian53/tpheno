# encoding: utf-8

import gensim
import os
import collections
import smart_open
import random


################################################################################################################
#  
# Reference 
# ---------
#    https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb
#   
#    * IMDB sentiment analysis 
#        https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb
# 
#
# 
# [Log]
# 1. Warning: 
#     anaconda/lib/python2.7/site-packages/gensim/utils.py:1015: UserWarning: Pattern library is not installed, 
#     lemmatization won't be available. warnings.warn("Pattern library is not installed, lemmatization won't be available.")
#
#
#

# Set file names for train and test data
test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
print('> test_data_dir: %s' % test_data_dir)  # .../anaconda/lib/python2.7/site-packages/gensim/test/test_data
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'

print('> train file: %s\n> test file:%s\n' % (lee_train_file, lee_test_file))


def read_corpus(fname, tokens_only=False):
    """

    Memo
    ----
    1. For a given file (aka corpus), each continuous line constitutes a single document and the length of each line (i.e., document) can vary.
    2. To train the model, we'll need to associate a tag/number with each document of the training corpus. In our case, the tag is simply the zero-based line number.
    """
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                # if i < 2: 
                #     print '  + simple preprocess> %s' % gensim.utils.simple_preprocess(line) # turns a string into a list of tokens
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])  ### 

def getTrainTest(**kargs):
    train_corpus = list(read_corpus(lee_train_file))
    test_corpus = list(read_corpus(lee_test_file, tokens_only=True)) 

    return (train_corpus, test_corpus)

def assess(**kargs): 
    return assessModel(**kargs)
def assessModel(model, train_corpus, test_corpus=None): # assess model 
    """
    Assess the model by self similarity. 

    To assess our new model, we'll first infer new vectors for each document of the training corpus, 
    compare the inferred vectors with the training corpus, and then returning the rank of the document based on self-similarity. 
    Basically, we're pretending as if the training corpus is some new unseen data and then seeing how they compare with the trained model.
    
    The expectation is that we've likely overfit our model (i.e., all of the ranks will be less than 2) and 
    so we should be able to find similar documents very easily. Additionally, we'll keep track of the second ranks 
    for a comparison of less similar documents.
    """
    ranks = []
    second_ranks = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        # if doc_id < 2: 
        #     print('  + sims (topn=%d): %s' % (len(model.docvecs), sims[:10]))
        
        rank = [docid for docid, sim in sims].index(doc_id)  # where is myself in this rank ordering? 
        ranks.append(rank)  # maps doc_id to its rank order
    
        second_ranks.append(sims[1])  # second most similar doc (ID, score)

    # print('  + second_ranks:\n%s\n' % second_ranks[:10])

    # result 
    print collections.Counter(ranks)  # Results vary due to random seeding and very small corpus

    # verify 
    print('\n  + what attributes does a tagged document have? %s\n\n' % dir(train_corpus[doc_id]))
    # [log] 'count', 'index', 'tags', 'words' ...

    print('Document ({}): <<{}>>\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

    # Pick a random document from the test corpus and infer a vector from the model
    doc_id = random.randint(0, len(train_corpus))

    # Compare and print the most/median/least similar documents from the train corpus
    print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
    sim_id = second_ranks[doc_id]  # (id, score)
    print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))

    return

def testModel(model, train_corpus, test_corpus): 
    """ 
    Using the same approach above, we'll infer the vector for a randomly chosen test document, and compare the document to our model by eye.
    """

    # Pick a random document from the test corpus and infer a vector from the model
    doc_id = random.randint(0, len(test_corpus))
    inferred_vector = model.infer_vector(test_corpus[doc_id])
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    # Compare and print the most/median/least similar documents from the train corpus
    print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
    
    return

def getDocVec(model, train_corpus, test_corpus):
    
    n_doc = len(model.docvecs)
    print('  + n_doc: %d' % n_doc)
    for i in range(n_doc): 
        if i % 100 == 0: 
            print(' i=%d => \n%s\n' % (i, model.docvecs[i]))
            print('      => \n%s\n' % model.docvecs[train_corpus[i].tags[0]])

    for doc_id in range(len(train_corpus)):
        assert all(model.docvecs[train_corpus[doc_id].tags[0]] == model.docvecs[doc_id])
       
    return 

def t_d2v(**kargs): 

    # get training and test data
    
    n_test = 5
    train_corpus = list(read_corpus(lee_train_file))
    test_corpus = list(read_corpus(lee_test_file, tokens_only=True)) 
    # for i, cor in enumerate(train_corpus):  
    #     print('  + [%d] \n%s\n' % (i, cor))
    #     if i >= n_test: break


    # train the model 
    # 1. init 
    model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)

    # 2. build vocabulary 
    model.build_vocab(train_corpus) # build dictionary of unique words + count (e.g. model.wv.vocab['penalty'].count for counts for the word penalty)

    # 3. train
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

    # 4. infer a vector given a sequence 
    model.infer_vector(['only', 'you', 'can', 'prevent', 'forrest', 'fires'])
    # This vector can then be compared with other vectors via cosine similarity.

    assess(model=model, train_corpus=train_corpus, test_corpus=test_corpus)  

    # get individual vector 
    getDocVec(model, train_corpus=train_corpus, test_corpus=test_corpus)  
    
    return

def test(**kargs): 
    
    t_d2v(**kargs)
    
    return

if __name__ == "__main__": 
    test()