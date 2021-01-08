from scipy import spatial
from sent2vec.vectorizer import Vectorizer
import gensim
from sent2vec.splitter import Splitter
from gensim.models.word2vec import Word2Vec
import numpy as np
from gensim.test.utils import common_texts
from gensim.models import KeyedVectors
# load word2vec model, here GoogleNews is used
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
itSolDfInputName = "1.list_it_solutions_modelInput.csv"
busNeedDfInputName = '2.list_business_needs_modelInput.csv'
model = gensim.models.KeyedVectors.load_word2vec_format(
    './GoogleNews-vectors-negative300.bin', binary=True)


def ITSolList(Bus_need_code):
    busNeed_df = pd.read_csv(busNeedDfInputName)
    busNeedModelInput = busNeed_df[
        (busNeed_df['Reference Code'] == Bus_need_code)]
    if len(busNeedModelInput) == 0:
        print('Business Need Ref Code Error')
        return []
    busNeedEnCode = busNeedModelInput['Model Input'].iloc[0]
    #print(busNeedEnCode)
    s1_afv = avg_feature_vector(
        busNeedEnCode, model=model, num_features=300, index2word_set=set(model.wv.index2word))
    itSol_df = pd.read_csv(itSolDfInputName)
    itSol_df = itSol_df.dropna(subset=['Model Input'])

    itSol_df['Cosine-Similarity'] = itSol_df.loc[:, 'Model Input'].apply(
        lambda x: similarity(s1_afv, avg_feature_vector(x, model=model,
                                                        num_features=300, index2word_set=set(model.wv.index2word))).item())
    # calculate distance between two sentences using WMD algorithm
    itSol_df = itSol_df.sort_values('Cosine-Similarity', ascending=False)
    itSol_df = itSol_df.loc[
        :, ["Reference Code", "Solution Name (Eng)", "Cosine-Similarity"]]
    itSol_df = itSol_df.iloc[:100]
    itSolOutputName = "ITSolList_" + Bus_need_code + "_" + ".csv"
    itSol_df.to_csv(itSolOutputName, index=False)
    # print('similarity = %.3f' % (distance))
    print("Finished IT solutions list:", itSolOutputName)


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


# print(len(s2_afv))


def similarity(s1, s2):
    return 1 - spatial.distance.cosine(s1, s2)


if __name__ == "__main__":
    bus_need_code = str(input("Enter the Business need code: "))
    ITSolList(bus_need_code)
