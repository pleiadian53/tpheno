#!/usr/bin/env python
# -*- coding: utf-8 -*-


############################################
#  Reference: gensim.word2vec
#
#####

MAX_WORDS_IN_BATCH = 20001 # stay consistent with tdoc.TDoc.vocab_size
try: 
    from tdoc import TDoc
    MAX_WORDS_IN_BATCH = TDoc.vocab_size # use tdoc.TDoc.vocab_size
except: 
    pass 

class LineSentence(object):
    """
    Simple format: one sentence = one line; words already preprocessed and separated by whitespace.
    """

    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None):
        """
        `source` can be either a string or a file object. Clip the file to the first
        `limit` lines (or no clipped if limit is None, the default).

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in itertools.islice(self.source, self.limit):
                line = utils.to_unicode(line).split()
                i = 0
                while i < len(line):
                    yield line[i : i + self.max_sentence_length]
                    i += self.max_sentence_length
        except AttributeError:
            
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for line in itertools.islice(fin, self.limit):
                    line = utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i : i + self.max_sentence_length]
                        i += self.max_sentence_length


def test(): 
    pass

if __name__ == "__main__": 
    test()
    

