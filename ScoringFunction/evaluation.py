import numpy as np
import pandas as pd


class Evaluator:
    def __init__(self, list_of_matching_file_name = "3.list_of_matching.xlsx",modelName="stsb-roberta-large", 
        numberOfWords = 10000, encoder = 'BiEncoder'):
        self.listOfMatching_df = pd.read_excel(list_of_matching_file_name)

        # for finding the ITsolList files
        self.modelName = modelName
        self.numberOfWords = numberOfWords
        self.encoder = encoder
        pass

    def evaluate(self,itSolOutputName=""):
        listOfMatching_eval_df = self.listOfMatching_df
        listOfMatching_eval_df = listOfMatching_eval_df.apply( lambda row: generateRankingSum(row['Needs Ref'], row['Solution Ref (1)'], row['Solution Ref (2)'], row['Solution Ref (3)'], row['Solution Ref (4)']) )
        
        if str(itSolOutputName) == 0:
            itSolOutputName = 'Eval_model_' + self.modelName.replace('/','-') + "_csv"
        listOfMatching_eval_df.to_csv(itSolOutputName,index=False)
    

    def generateRankingSum(self, businessCode, solutionRef_1, solutionRef_2, solutionRef_3, solutionRef_4):
        ITsolListFileName = "ITSolList_"+busNeedCode+"_model_"+ self.modelName.replace('/','-') +
            '_numberOfWords_'+ str(self.numberOfWords) +"_"+ self.encoder+ ".csv"
        ITsolList_df = pd.read_csv(ITsolListFileName)
        
        Sum = findITsolRank(ITsolList_df, solutionRef_1)
        Sum += findITsolRank(ITsolList_df, solutionRef_2)
        Sum += findITsolRank(ITsolList_df, solutionRef_3)
        Sum += findITsolRank(ITsolList_df, solutionRef_4)

        return Sum
        

    def findITsolRank(ITsolList_df, solutionRef):
        if pd.isnull(solutionRef) is False:
            return 0
        if str(solutionRef)[:2] != "S-": 
            return 0
        return (ITsolList_df["Reference Code"] == solutionRef).idxmax() + 1
        


        

        


if __name__ == '__main__':
    ev = Evaluator()
    ev.evaluate()
    print(ev.listOfMatching_df.columns)