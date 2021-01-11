import numpy as np
import pandas as pd
import time # for timing

from sentence_transformers import SentenceTransformer, util, CrossEncoder
pd.set_option("display.max_rows", None, "display.max_columns", None)

#from transformers import LongformerModel

class ITsolListGenerator_BiEncoder:
    """A class used to generate resultant IT solution List of a specific business needs used. 
        It used sBert BiEnoder Model for ITsolList Generation
    To use:
    >>> itg_be = ITsolListGenerator_BiEncoder(modelName="stsb-roberta-large")
    >>> itg_be.generateModelEmbedding(sent_length=50)
    >>> itg_be.generateITSolList(bus'N-0001', sent_length=10000)
  
    
    Args:
        modelName (str, optional): sBERT model name used for generate the IT solution List. Defaults to 'distilbert-base-nli-stsb-mean-tokens'.
        sent_length (int, optional): The number of words for the modelInputs of each IT solutions/ business needs. Defaults to 100000.
    """
  
    def __init__(self, modelName='textattack/xlnet-large-cased-STS-B', sent_length = 100000):
        self._modelName = modelName
        self._model = SentenceTransformer(modelName)
        self._sent_length = sent_length

    def generateModelEmbedding(self, sent_length = 100000, itSolDfInputName="1.list_it_solutions_modelInput.csv", 
        itSolDfOutputName="", busNeedDfInputName='2.list_business_needs_modelInput.csv', busNeedDfOutputName=""):
        """To generate Model Embedding from String model input, The Model Embedding is saved as
        1.list_it_solutions_modelEmbedding.pkl and 2.list_business_needs_modelEmbedding.pkl
        Args:
            sent_length (int, optional): The number of words for the modelInputs of each IT solutions/ business needs. Defaults to 100000.
            itSolDfInputName (str, optional): The input CSV File Name of the IT Solution List.  Defaults to "1.list_it_solutions_modelInput.csv".
            itSolDfOutputName (str, optional): The output pickle File Name of the IT Solution List with "Model Embedding" columns. Defaults to "".
            busNeedDfInputName (str, optional): The input CSV File Name of the Business Need List. Defaults to '2.list_business_needs_modelInput.csv'.
            busNeedDfOutputName (str, optional): The output pickle File Name of the Business Need List with "Model Embedding" columns. . Defaults to "".
        """

        self._sent_length = sent_length

        # generate business needs Model Word Embedding
        busNeed_df = pd.read_csv(busNeedDfInputName)
        busNeed_df = busNeed_df.dropna(subset=['Model Input'])
        # generate Embedding from "Model Input" columns using the model.encode() function
        busNeed_df['Model Embedding'] = busNeed_df.loc[:,"Model Input"].apply( lambda x: self._model.encode( ' '.join(x.split(' ')[:sent_length] )) )
        if len(str(busNeedDfOutputName)) == 0: 
            busNeedDfOutputName = '2.list_business_needs_modelEmbedding_' + str(sent_length) + 'sl.pkl'
        busNeed_df.to_pickle(busNeedDfOutputName)  # save it with file name busNeedDfOutputName

        # generate IT solution Model Word Embedding
        itSol_df = pd.read_csv(itSolDfInputName)
        itSol_df = itSol_df.dropna(subset=['Model Input'])
        # generate Embedding from "Model Input" columns using the model.encode() function
        itSol_df['Model Embedding'] = itSol_df.loc[:,"Model Input"].apply( lambda x: self._model.encode( ' '.join(x.split(' ')[:sent_length] )) )
        if len(str(itSolDfOutputName)) == 0: 
            itSolDfOutputName = '1.list_it_solutions_modelEmbedding_' + str(sent_length) + 'sl.pkl'
        itSol_df.to_pickle(itSolDfOutputName) # save it with file name itSolDfOutputName
 

    def generateITSolList(self, busNeedCode, itSolListSize = 300, sent_length = 0, itSolDfInputName="", 
        busNeedDfInputName="", itSolOutputName="" ):
        """To generate the ITSolList, print it and save it in "ITSolList-<busNeedCode>.csv"
        Args:
            busNeedCode (int): The ID Code for the business need needed to be generated a IT solution list. eg. "N-0001"
            itSolListSize (int, optional): IT solution list size. Defaults to 300.
            sent_length (int, optional): number of sentence length of modelEmbedding to use for generating ITSolList. Defaults to 0.
            itSolDfInputName (str, optional): The input pickle File Name of the IT Solution List with "Model Embedding" columns. (file generated from function self.generateModelEmbedding) Defaults to "".
            busNeedDfInputName (str, optional): The input pickle File Name of the business needs with "Model Embedding" columns. (file generated from function self.generateModelEmbedding) Defaults to "".
            itSolOutputName (str, optional): The output csv file name of the generated ranked IT Solution List of the business need (specified by busNeedCode ) . Defaults to "".
        Returns:
            DataFrame: The pandas DataFrame of the generated ranked IT Solution List of the business need (specified by busNeedCode )
        """

        # Arguments input checking (use default value if the input invalid)
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
            itSolOutputName = "ITSolList_"+busNeedCode+"_"+self._modelName.split('/')[-1] +'_'+ str(sent_length) +"sl_BiEncoder.csv"
        itSol_df.to_csv(itSolOutputName,index=False)
        print("Finished IT solutions list:", itSolOutputName)
        return itSol_df

class ITsolListGenerator_CrossEncoder:
    """A class used to generate resultant IT solution List of a specific business needs used. 
        It used sBert CrossEncoder Model for ITsolList Generation
    
    Args:
            modelName (str, optional): sBERT model name used for generate the IT solution List. Defaults to 'cross-encoder/stsb-roberta-base'.
    """
    def __init__(self, modelName='cross-encoder/stsb-roberta-base'):
        self._modelName = modelName
        self._model = CrossEncoder(modelName)

    def generateITSolList(self, busNeedCode, itSolListSize = 400, itSolDfInputName="1.list_it_solutions_modelInput.csv", 
        busNeedDfInputName='2.list_business_needs_modelInput.csv', itSolOutputName='ITSolList.csv' ):
        """To generate the ITSolList, save it in "ITSolList-<busNeedCode>.csv"
        Args:
            busNeedCode (int): The ID Code for the business need needed to be generated a IT solution list. eg. "N-0001"
            itSolListSize (int, optional): The generate IT Solution List size. Defaults to 400.
            itSolDfInputName (str, optional): The input CSV File Name of the IT Solution List.  Defaults to "1.list_it_solutions_modelInput.csv".
            busNeedDfInputName (str, optional): The input CSV File Name of the Business Need List. Defaults to '2.list_business_needs_modelInput.csv'.
            itSolOutputName (str, optional): The output csv file name of the generated ranked IT Solution List of the business need (specified by busNeedCode ) . Defaults to 'ITSolList.csv'.
        """

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

# roberta fine tuned with stsb
#itg_be = ITsolListGenerator_BiEncoder(modelName="stsb-roberta-large")

# longformer no fine tuning
#itg_be = ITsolListGenerator_BiEncoder(modelName="allenai/longformer-base-4096")

# XLNet fine tuned with stsb
itg_be = ITsolListGenerator_BiEncoder(modelName="textattack/xlnet-large-cased-STS-B")
itg_be.generateModelEmbedding()

itg_be.generateITSolList('N-0010', sent_length=100000)
itg_be.generateITSolList('N-0011', sent_length=100000)
itg_be.generateITSolList('N-0012', sent_length=100000)
itg_be.generateITSolList('N-0013', sent_length=100000)
itg_be.generateITSolList('N-0014', sent_length=100000)
itg_be.generateITSolList('N-0015', sent_length=100000)
itg_be.generateITSolList('N-0016', sent_length=100000)



# itg_ce = ITsolListGenerator_CrossEncoder()
# itg_ce.generateITSolList('N-0003')


t1 = time.time()
print("Time Used (s):",t1-t0)

# print("max length:",itg_be._model.max_seq_length) #128