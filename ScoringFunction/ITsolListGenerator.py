import numpy as np
import pandas as pd
import time # for timing

from sentence_transformers import SentenceTransformer, util, CrossEncoder
pd.set_option("display.max_rows", None, "display.max_columns", None)


class ITsolListGenerator_BiEncoder:
    '''ITsolListGenerator BiEncoder 
    '''
    
    def __init__(self, modelName='distilbert-base-nli-stsb-mean-tokens', sent_length = 100000):
        self._modelName = modelName
        self._model = SentenceTransformer(modelName)
        self._sent_length = sent_length

    def generateModelEmbedding(self, sent_length = 100000, itSolDfInputName="1.list_it_solutions_modelInput.csv", 
        itSolDfOutputName="", busNeedDfInputName='2.list_business_needs_modelInput.csv', busNeedDfOutputName=""):
        '''To generate Model Embedding from String model input, 
            The Model Embedding is saved in 
            1.list_it_solutions_modelEmbedding.pkl and 2.list_business_needs_modelEmbedding.pkl
            Arg:
            sent_length (int): number of words maximum in model Input
        '''
        self._sent_length = sent_length

        busNeed_df = pd.read_csv(busNeedDfInputName)
        busNeed_df = busNeed_df.dropna(subset=['Model Input'])
        busNeed_df['Model Embedding'] = busNeed_df.loc[:,"Model Input"].apply( lambda x: self._model.encode( ' '.join(x.split(' ')[:sent_length] )) )
        if len(str(busNeedDfOutputName)) == 0: 
            busNeedDfOutputName = '2.list_business_needs_modelEmbedding_' + str(sent_length) + 'sl.pkl'
        busNeed_df.to_pickle(busNeedDfOutputName)

        itSol_df = pd.read_csv(itSolDfInputName)
        itSol_df = itSol_df.dropna(subset=['Model Input'])
        itSol_df['Model Embedding'] = itSol_df.loc[:,"Model Input"].apply( lambda x: self._model.encode( ' '.join(x.split(' ')[:sent_length] )) )
        if len(str(itSolDfOutputName)) == 0: 
            itSolDfOutputName = '1.list_it_solutions_modelEmbedding_' + str(sent_length) + 'sl.pkl'
        itSol_df.to_pickle(itSolDfOutputName)

    def generateITSolList(self, busNeedCode, itSolListSize = 300, sent_length = 0, itSolDfInputName="", 
        busNeedDfInputName="", itSolOutputName="" ):
        '''To generate the ITSolList, print it and save it in "ITSolList-<busNeedCode>.csv"
        Args:
        busNeedCode (str): eg. "N-0001"
        itSolListSize (int): IT solution list size (default: 300)
        sent_length (int): number of sentence length of modelEmbedding to use for generating ITSolList
        itSolDfInputName (str)
        busNeedDfInputName (str)
        '''
        if sent_length == 0:
            sent_length = self._sent_length

        if len(str(busNeedDfInputName)) == 0: 
            busNeedDfInputName = '2.list_business_needs_modelEmbedding_' + str(sent_length) + 'sl.pkl'

        if len(str(itSolDfInputName)) == 0: 
            itSolDfInputName = '1.list_it_solutions_modelEmbedding_' + str(sent_length) + 'sl.pkl'

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
        if len(str(itSolOutputName)) == 0:
            itSolOutputName = "ITSolList_"+busNeedCode+"_"+self._modelName +'_'+ str(sent_length) +"sl_BiEncoder.csv"
        itSol_df.to_csv(itSolOutputName,index=False)
        print("Finished IT solutions list:", itSolOutputName)

class ITsolListGenerator_CrossEncoder:
    '''CrossEncoder
    '''
    def __init__(self, modelName='cross-encoder/stsb-roberta-base'):
        self._modelName = modelName
        self._model = CrossEncoder(modelName)

    def generateITSolList(self, busNeedCode, itSolListSize = 400, itSolDfInputName="1.list_it_solutions_modelInput.csv", 
        busNeedDfInputName='2.list_business_needs_modelInput.csv', itSolOutputName='ITSolList.csv' ):
        '''To generate the ITSolList, print it and save it in "ITSolList-<busNeedCode>.csv"
        Args:
        busNeedCode (str): eg. "N-0001"
        itSolListSize (int): IT solution list size (default: 400)
        itSolDfInputName (str)
        busNeedDfInputName (str)
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

# itg_be.generateModelEmbedding(sent_length=50)

itg_be.generateITSolList('N-0001', sent_length=10000)
itg_be.generateITSolList('N-0002', sent_length=10000)
itg_be.generateITSolList('N-0003', sent_length=10000)
itg_be.generateITSolList('N-0004', sent_length=10000)



# itg_ce = ITsolListGenerator_CrossEncoder()
# itg_ce.generateITSolList('N-0003')


t1 = time.time()
print("Time Used (s):",t1-t0)

# print("max length:",itg_be._model.max_seq_length) #128
