import numpy as np
import pandas as pd
import time


class Evaluator:
    def __init__(self, list_of_matching_file_name = "3.list_of_matching.xlsx",modelName="stsb-roberta-large", 
        numberOfWords = 100000, islemmatize = False,encoder = 'BiEncoder', directory = ""):
        self.listOfMatching_df = pd.read_excel(list_of_matching_file_name)

        # for finding the ITsolList files
        self.modelName = modelName
        self.modelNameColumn = modelName.replace('/','-')
        self.modelNameString = "_model_" +modelName.replace('/','-')
        self.numberOfWords = numberOfWords
        self.numberOfWordsString = "_numberOfWords_"+ str(numberOfWords) if encoder == 'BiEncoder' else ""
        self.encoder = encoder
        self.encoderString = "_" + encoder
        self.islemmatizeString = "_lemmatize" if islemmatize else "_notLemmatize"

        self.directory = directory
        

    def evaluate(self,evalResultOutputName=""):
        listOfMatching_eval_df = self.listOfMatching_df
        listOfMatching_eval_df[self.modelNameColumn] = listOfMatching_eval_df.apply( lambda row: self.generateRankingSum(row['Needs Ref'], row['Solution Ref (1)'], row['Solution Ref (2)'], row['Solution Ref (3)'], row['Solution Ref (4)']), axis=1 )
        
        if len(evalResultOutputName) == 0:
            evalResultOutputName = 'Eval' + self.modelNameString + self.numberOfWordsString+self.islemmatizeString +self.encoderString + ".csv"
        listOfMatching_eval_df.to_csv(evalResultOutputName,index=False)
    

    def generateRankingSum(self, busNeedCode, solutionRef_1, solutionRef_2, solutionRef_3, solutionRef_4):
        ITsolListFileName = self.directory+ "ITSolList_"+busNeedCode+ self.modelNameString+ self.numberOfWordsString + self.islemmatizeString + self.encoderString+ ".csv"
        ITsolList_df = pd.read_csv(ITsolListFileName)
        
        Sum = self.findITsolRank(ITsolList_df, solutionRef_1)
        Sum += self.findITsolRank(ITsolList_df, solutionRef_2)
        Sum += self.findITsolRank(ITsolList_df, solutionRef_3)
        Sum += self.findITsolRank(ITsolList_df, solutionRef_4)
        return Sum
        

    def findITsolRank(self,ITsolList_df, solutionRef):
        if pd.isnull(solutionRef):
            return 0
        if str(solutionRef)[:2] != "S-": 
            return 0
        return (ITsolList_df["Reference Code"] == solutionRef).idxmax() + 1
        


        

        


if __name__ == '__main__':
    t0 = time.time()
    # ev = Evaluator(directory='OutputBackup/stsb-roberta-large_notLemmatize/', encoder='BiEncoder', modelName='stsb-roberta-large',islemmatize=False)
    ev = Evaluator(directory='', encoder='BiEncoder', modelName='bert-large-nli-stsb-mean-tokens',islemmatize=False)
    ev.evaluate()
    t1 = time.time()
    print("Time Used (s):",t1-t0)
    