# ==============================================================================
#  
# Objective: Test Gensim implementation of doc2vec models (e.g. skip-gram)
# 
# Reference: The TensorFlow Authors. 
#
#
# ==============================================================================


from gensim.models import doc2vec
from collections import namedtuple

import csv
import re
import string

import sys, os 

# local modules (assuming that PYTHONPATH has been set)
from batchpheno import icd9utils, sampling, qrymed2
from config import seq_maker_config, sys_config
from batchpheno.utils import div
from pattern import medcode as pmed
from seqmaker import seqparams
from seqmaker import analyzer
from seqmaker import seqAnalyzer as sa 


# We use the wikipedia dataset to train the paragraph vector
# reader = csv.reader(open("wikipedia.csv")) # NLP 

basedir = sys_config.read('DataExpRoot')
docfile = 'condition_drug_seq.dat'
ipath = os.path.join(basedir, docfile)

# reader = csv.reader(open(ipath))
# count = 0
# data = ''
# for row in reader:
#     count = count + 1
#     if count > 301:
#        break
#     else:
#         # data += row[1]
#         # print('row> %s' % row)
#         data += row[1]

# # Setup a regex to split paragraph to sentences. 
# # We assume sentences ending with . ? or !. There are sentences that
# # have . in the middle such as Mr.Wang, as the majoirty sentences are
# # okay and this program is mostly serve as a demo, I decide to ignore 
# # these cases. 
# sentenceEnders = re.compile('[#]') # in regular NLP would be: [.?!]
# data_list = sentenceEnders.split(data)  # list of strings/sentences

# read(load_=load_seq, simplify_code=simplify_code, mode=read_mode)
sequences = data_list = sa.read(mode='doc')  # e.g. max_doc=10, 10 patiets => 300+ sentences

print('verify> show example doc sequences (size: %d) ...' % len(data_list))
for i, seq in enumerate(data_list): 
    if i > 50: break
    print("> sequence #%d:\n%s\n" % (i+1, seq))

# I created a namedtuple with words=['I', 'love', 'NLP'] and tags=['SEN_1']
# to represent an input sentence
LabelDoc = namedtuple('LabelDoc', 'words tags')
exclude = set(string.punctuation)
all_docs = []
count = 0
for sen in data_list:
    if isinstance(sen, str): 
        word_list = sen.split() 
    else: 
        word_list = sen  # split is already done

    # For every sentences, if the length is less than 3, we may want to discard it
    # as it seems too short. 
    # if len(word_list) < 3: continue   # filter short sentences
    
    tag = ['SEN_' + str(count)]
    count += 1
    if isinstance(sen, str): 
        sen = ''.join(ch for ch in sen if ch not in exclude)  # filter excluded characters
        all_docs.append(LabelDoc(sen.split(), tag))
    else:  
        all_docs.append(LabelDoc(sen, tag)) # assuming unwanted char already filetered 
    

# Print out a sample for one to view what the structure is looking like    
# print all_docs[0:300]
for i, doc in enumerate(all_docs[0:120]): 
    print('> doc #%d: %s' % (i, doc))

# sys.exit(0)

# Mikolov pointed out that to reach a better result, you may either want to shuffle the 
# input sentences or to decrease the learning rate alpha. We use the latter one as pointed
# out from the blog provided by http://rare-technologies.com/doc2vec-tutorial/
model = doc2vec.Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
model.build_vocab(all_docs)
for epoch in range(10):
    model.train(all_docs)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay    

# Finally, we save the model
descriptor = 'pm'
doctype = 'doc2vec' 
doc_basename = 'condition_drug'
if descriptor is None: descriptor = 'test'
model.save('%s_%s.%s' % (doc_basename, descriptor, doctype))


