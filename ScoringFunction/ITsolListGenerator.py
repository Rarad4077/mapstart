import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer, util
pd.set_option("display.max_rows", None, "display.max_columns", None)


class ITsolListGenerator:
    
    def __init__(self, modelName='distilbert-base-nli-stsb-mean-tokens'):
        self._modelName = modelName
        self._model = SentenceTransformer(modelName)

    def generateModelEmbedding(self,
        itSolDfInputName="1.list_it_solutions_modelInput.csv", 
        itSolDfOutputName="1.list_it_solutions_modelEmbedding.pkl",
        busNeedDfInputName='2.list_business_needs_modelInput.csv',
        busNeedDfOutputName='2.list_business_needs_modelEmbedding.pkl'):
        '''To generate Model Embedding from String model input, 
            The Model Embedding is saved in 
            1.list_it_solutions_modelEmbedding.pkl and 2.list_business_needs_modelEmbedding.pkl
        '''

        busNeed_df = pd.read_csv(busNeedDfInputName)
        busNeed_df = busNeed_df.dropna(subset=['Model Input'])
        busNeed_df['Model Embedding'] = busNeed_df.loc[:,"Model Input"].apply( lambda x: self._model.encode(x))
        busNeed_df.to_pickle(busNeedDfOutputName)

        itSol_df = pd.read_csv(itSolDfInputName)
        itSol_df = itSol_df.dropna(subset=['Model Input'])
        itSol_df['Model Embedding'] = itSol_df.loc[:,"Model Input"].apply( lambda x: self._model.encode(x))
        itSol_df.to_pickle(itSolDfOutputName)




    def generateITSolList(self, 
        busNeedCode,
        itSolDfInputName="1.list_it_solutions_modelEmbedding.pkl", 
        busNeedDfInputName='2.list_business_needs_modelEmbedding.pkl', 
        itSolOutputName='ITSolList.csv' ):
        '''To generate the ITSolList, print it and save it in "ITSolList-<busNeedCode>.csv"
        '''

        # Extract the business need emnbedding by the busNeedCode
        busNeed_df = pd.read_pickle(busNeedDfInputName)
        busNeedModelInput = busNeed_df[(busNeed_df['Reference Code'] == busNeedCode)]
        if len(busNeedModelInput) == 0:
            print('Business Need Ref Code Error')
            return []
        # print(busNeedModelInput['Model Embedding'].iloc[0])
        busNeedEnCode = busNeedModelInput['Model Embedding'].iloc[0]

        #generate the cosine simlarity
        itSol_df = pd.read_pickle(itSolDfInputName)
        itSol_df['Cosine Similarity'] = itSol_df.loc[:,"Model Embedding"].apply(
            lambda x: util.pytorch_cos_sim(busNeedEnCode, x).item())
        itSol_df = itSol_df.sort_values(by='Cosine Similarity', ascending=False)

        itSol_df = itSol_df.loc[:,["Reference Code","Solution Name (Eng)","Cosine Similarity"]]

        # print(itSol_df)
        itSolOutputName = "ITSolList_"+busNeedCode+"_"+self._modelName +".csv"
        itSol_df.to_csv(itSolOutputName,index=False)
        print("Finished IT solutions list:", itSolOutputName)
        



    
itg = ITsolListGenerator(modelName="stsb-roberta-large")

# itg.generateModelEmbedding()
itg.generateITSolList('N-0025')
