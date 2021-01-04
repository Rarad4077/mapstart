import numpy as np
import pandas as pd
import time # for timing

from sentence_transformers import SentenceTransformer, util, CrossEncoder
pd.set_option("display.max_rows", None, "display.max_columns", None)


class ITsolListGenerator_BiEncoder:
    
    def __init__(self, modelName='distilbert-base-nli-stsb-mean-tokens'):
        self._modelName = modelName
        self._model = SentenceTransformer(modelName)

    def generateModelEmbedding(self,
        sent_length = 10000,  
        itSolDfInputName="1.list_it_solutions_modelInput.csv", 
        itSolDfOutputName="1.list_it_solutions_modelEmbedding.pkl",
        busNeedDfInputName='2.list_business_needs_modelInput.csv',
        busNeedDfOutputName='2.list_business_needs_modelEmbedding.pkl'):
        '''To generate Model Embedding from String model input, 
            The Model Embedding is saved in 
            1.list_it_solutions_modelEmbedding.pkl and 2.list_business_needs_modelEmbedding.pkl
            Arg:
            sent_length (int): number of words maximum in model Input
        '''

        busNeed_df = pd.read_csv(busNeedDfInputName)
        busNeed_df = busNeed_df.dropna(subset=['Model Input'])
        busNeed_df['Model Embedding'] = busNeed_df.loc[:,"Model Input"].apply( lambda x: self._model.encode( ' '.join(x.split(' ')[:sent_length] )) )
        busNeed_df.to_pickle(busNeedDfOutputName)

        itSol_df = pd.read_csv(itSolDfInputName)
        itSol_df = itSol_df.dropna(subset=['Model Input'])
        itSol_df['Model Embedding'] = itSol_df.loc[:,"Model Input"].apply( lambda x: self._model.encode( ' '.join(x.split(' ')[:sent_length] )) )
        itSol_df.to_pickle(itSolDfOutputName)

    def generateITSolList(self, 
        busNeedCode,
        itSolListSize = 300,
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
        itSol_df = itSol_df.iloc[:itSolListSize]
        # print(itSol_df)
        itSolOutputName = "ITSolList_"+busNeedCode+"_"+self._modelName +"_BiEncoder.csv"
        itSol_df.to_csv(itSolOutputName,index=False)
        print("Finished IT solutions list:", itSolOutputName)

class ITsolListGenerator_CrossEncoder:
    '''CrossEncoder
    '''
    def __init__(self, modelName='cross-encoder/stsb-roberta-base'):
        self._modelName = modelName
        self._model = CrossEncoder(modelName)

    def generateITSolList(self, 
        busNeedCode,
        itSolListSize = 400,
        itSolDfInputName="1.list_it_solutions_modelInput.csv", 
        busNeedDfInputName='2.list_business_needs_modelInput.csv', 
        itSolOutputName='ITSolList.csv' ):
        '''To generate the ITSolList, print it and save it in "ITSolList-<busNeedCode>.csv"
        '''

        busNeed_df = pd.read_csv(busNeedDfInputName)
        busNeed_df = busNeed_df.dropna(subset=['Model Input'])
        busNeedModelInput = busNeed_df[(busNeed_df['Reference Code'] == busNeedCode)]
        if len(busNeedModelInput) == 0:
            print('Business Need Ref Code Error')
            return []
        busNeedEnCode = busNeedModelInput['Model Input'].iloc[0]

        itSol_df = pd.read_csv(itSolDfInputName)
        itSol_df = itSol_df.dropna(subset=['Model Input'])

        busNeedEnCodeList = [busNeedEnCode for _ in range(len(itSol_df)) ]
        itSolEnCodeList = list(itSol_df.loc[:,"Model Input"])
        predictions = [ (x,y) for x,y in zip(busNeedEnCodeList,itSolEnCodeList)]
        predictions = self._model.predict(predictions)

        print(predictions)

        itSol_df['Cosine Similarity'] = predictions

        itSol_df = itSol_df.loc[:,["Reference Code","Solution Name (Eng)","Cosine Similarity"]]
        itSol_df = itSol_df.iloc[:itSolListSize]
        itSol_df = itSol_df.sort_values(by='Cosine Similarity', ascending=False)
        # # print(itSol_df)
        itSolOutputName = "ITSolList_"+busNeedCode+"_"+self._modelName.split('/')[-1] +"_CrossEncoder.csv"
        itSol_df.to_csv(itSolOutputName,index=False)
        print("Finished IT solutions list:", itSolOutputName)



t0 = time.time()
    
itg_be = ITsolListGenerator_BiEncoder(modelName="stsb-roberta-large")

# itg_be.generateModelEmbedding(sent_length=1000)
# itg_be.generateITSolList('N-0026')
print("max length:",itg_be._model.max_seq_length) #128


# itg_ce = ITsolListGenerator_CrossEncoder()

# print("max length:",itg_ce._model.max_seq_length) #128

# itg_ce.generateITSolList('N-0003')


t1 = time.time()
print("Time Used:",t1-t0)
