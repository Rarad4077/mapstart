from scipy import spatial
from sent2vec.vectorizer import Vectorizer
import gensim
from sent2vec.splitter import Splitter
from gensim.models.word2vec import Word2Vec
import numpy as np
from gensim.test.utils import common_texts
from gensim.models import KeyedVectors
# load word2vec model, here GoogleNews is used
model = gensim.models.KeyedVectors.load_word2vec_format(
    './GoogleNews-vectors-negative300.bin', binary=True)
# two sample sentences
s1 = "man"
s2 = "man"


# calculate distance between two sentences using WMD algorithm
distance = model.similarity(s1, s2)

print('similarity = %.3f' % (1 - distance))
