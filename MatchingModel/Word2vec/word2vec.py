from scipy import spatial
from sent2vec.vectorizer import Vectorizer

from sent2vec.splitter import Splitter
from gensim.models.word2vec import Word2Vec
import numpy as np
sentences = [
    "Alice is in the Wonderland.",
    "Alice is not in the Wonderland.",
]

bn1 = "blockchain"
it1 = "Founded in 2001, Shanghai Xiaoi Robot Technology Co. Ltd is a leading AI company that specializes in multilingual natural language processing, deep semantic interaction, speech recognition, machine learning and other cognitive intelligence technologies. Its self-patented NLP-powered Chatbot Solutions possess the following capabilities. We have ample use cases across various business domains and public sectors, with solid track record in deploying solutions and applications to banking & finance, insurance, healthcare, education, transportation, e-commerce, FMCG, utilities, infrastructure and government projects etc."
it2 = "Blockchain technology is not only a platform on which the mass of new data derived from smart cities can be safely stored and accessed by those who should have access to it. The chain also may serve as the interoperable platform that gives residents of smart cities greater say in the decisions affecting their hyper-local communities, from budgeting to elections, etc. It may also serve as a reputation management tool, as these cities tend to be chock-full of citizens who demand a certain standard from individuals and businesses when it comes to communal and environmental care."


splitter = Splitter()
splitter.sent2words(sentences=sentences, remove_stop_words=[
                    'not'], add_stop_words=[])
print(splitter.words)

'''
vectorizer = Vectorizer()
vectorizer.word2vec(
    splitter.words, pretrained_vectors_path='./GoogleNews-vectors-negative300.bin')
vectors_w2v = vectorizer.vectors
dist_w2v = spatial.distance.cosine(vectors_w2v[0], vectors_w2v[1])
print('dist_w2v: {}'.format(dist_w2v))
'''
''''index2word_set = set(models.wv.index2word)

word2vec_model = gensim.models.Word2Vec(sentences, size=100, min_count=5)
bin_file = "vecmodel.csv"
word2vec_model.wv.save_word2vec_format(bin_file, binary=False)


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
    'this is a sentence', model=model, num_features=300, index2word_set=index2word_set)
s2_afv = avg_feature_vector('this is also sentence', model=model,
                            num_features=300, index2word_set=index2word_set)
sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
print(sim)'''
