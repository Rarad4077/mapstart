from scipy import spatial
from sent2vec.vectorizer import Vectorizer
import gensim
from sent2vec.splitter import Splitter
from gensim.models.word2vec import Word2Vec
import numpy as np
from gensim.test.utils import common_texts
from gensim.models import KeyedVectors


#DataSet#
sent1 = 'blockchain tech'
sent2 = 'Blockchain technology'
sentences = sent1 + sent2

#''''Applying Word2vec''''#
'''word2vec_model = gensim.models.Word2Vec(sentences, size=100, min_count=5)
bin_file = "vecmodel.csv"
word2vec_model.wv.save_word2vec_format(bin_file, binary=False)'''
model = gensim.models.KeyedVectors.load_word2vec_format(
    './GoogleNews-vectors-negative300.bin', binary=True)


def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

s1_afv = avg_feature_vector(
    sent1, model=model, num_features=300, index2word_set=set(model.wv.index2word))
# print(len(s1_afv))
s2_afv = avg_feature_vector(sent2, model=model,
                            num_features=300, index2word_set=set(model.wv.index2word))

# print(len(s2_afv))
sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
print('similarity = %.3f' % (sim))
