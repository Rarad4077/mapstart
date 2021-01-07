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
itSolDfInputName="1.list_it_solutions_modelEmbedding.pkl", 
busNeedDfInputName='2.list_business_needs_modelEmbedding.pkl'
model = gensim.models.KeyedVectors.load_word2vec_format(
        './GoogleNews-vectors-negative300.bin', binary=True)
def ITSolList(Bus_need_code)：
	busNeed_df = pd.read_pickle(busNeedDfInputName)
    busNeedModelInput = busNeed_df[(busNeed_df['Reference Code'] == busNeedCode)]
    if len(busNeedModelInput) == 0:
        print('Business Need Ref Code Error')
        return []
    busNeedEnCode = busNeedModelInput['Model Embedding'].iloc[0]
    itSol_df = pd.read_pickle(itSolDfInputName)
    itSol_df['WMD distance'] = itSol_df.loc[:,"Model Embedding"].apply(
    lambda x: model.similarity(busNeedEnCode, x).item())
    # calculate distance between two sentences using WMD algorithm
    itSol_df = itSol_df.sort_values(by='WMD distance', ascending=False)
    itSol_df = itSol_df.loc[:,["Reference Code","Solution Name (Eng)","WMD distance"]]
    itSol_df = itSol_df.iloc[:100]
    itSolOutputName = "ITSolList_"+busNeedCode+"_"+".csv"
    itSol_df.to_csv(itSolOutputName,index=False)
    #print('similarity = %.3f' % (distance))
    print("Finished IT solutions list:", itSolOutputName)

if __name__ == "__main__":
    bus_need_code = str(input("Enter the Business need code")) 
    ITSolList(bus_need_code)
