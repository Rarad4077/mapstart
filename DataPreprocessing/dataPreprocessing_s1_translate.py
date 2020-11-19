# !conda activate fypws1
import numpy as np
import pandas as pd
# !pip install langdetect
# from langdetect import detect
# pd.set_option('display.max_colwidth', -1)
# !pip install git+https://github.com/BoseCorp/py-googletrans.git --upgrade
from googletrans import Translator

class CustomTranslator:
  _translator = Translator()
  def __init__(self):
    pass
  
  def translateToEn(self,s):
    result = None
    print("translate to en:",  s[:10], end=' ')
    for _ in range(10):
      try:
        print(_,end=' ')
        result = self._translator.translate(s,dest="en")
        break
      except:
        self._translator = Translator()
    print("")
    if result is None:
      return s
    return result.text
  
  def translateITSolToEn(self, inputName='1.list_it_solutions.xlsx', outputName='1.list_it_solutions_clean.csv'):
    itSol_df = pd.read_excel(inputName)
    itSol_df.loc[:,"Solution Description"] = itSol_df.loc[:,"Solution Description"].apply(lambda x: self.translateToEn(x))
    itSol_df.loc[:,'Use Case'] = itSol_df.loc[:,'Use Case'].apply(lambda x: ct.translateToEn(x))
    itSol_df.to_csv(outputName)
  
  def translateBusNeedToEn(self,inputName, outputName):
    pass


ct = CustomTranslator()
# ct.translateITSolToEn()





