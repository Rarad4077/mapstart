# !conda activate fypws1
import numpy as np
import pandas as pd

# ----- for tranlate -----
# !pip install langdetect
# from langdetect import detect
# pd.set_option('display.max_colwidth', -1)
# !pip install git+https://github.com/BoseCorp/py-googletrans.git --upgrade
from googletrans import Translator

# ----- for lemmatizing -----
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet



class CustomTranslator:
  '''
  This class is used to translate the IT solution columns "Solution Description" and "Use Case" from TC to ENG
  '''
  _translator = Translator()
  def __init__(self):
    pass
  
  def translateToEn(self,s):
    if not s:
      return s
    result = None
    print("translate to en:",  str(s)[:10], end=' ')
    for _ in range(5):
      try:
        print(_,end=' ')
        result = self._translator.translate(s,dest="en")
        break
      except (AttributeError, TypeError):
        self._translator = Translator()

    print("")
    if result is None:
      return s
    return result.text
  
  def translateITSolToEn(self, inputName='1.list_it_solutions.xlsx', outputName='1.list_it_solutions_clean.csv'):
    itSol_df = pd.read_excel(inputName)
    itSol_df.loc[:,"Solution Name (Eng)"] = itSol_df.loc[:,"Solution Name (Eng)"].apply(lambda x: self.translateToEn(x))
    itSol_df.loc[:,"Solution Description"] = itSol_df.loc[:,"Solution Description"].apply(lambda x: self.translateToEn(x))
    itSol_df.loc[:,'Use Case'] = itSol_df.loc[:,'Use Case'].apply(lambda x: ct.translateToEn(x))
    itSol_df.to_csv(outputName, index=False)
  
  def translateBusNeedToEn(self,inputName='2.list_business_needs.xlsx', outputName='2.list_business_needs_clean.csv'):
    ''' Bus Need already have ENG and TC col, which do not need to translate
    '''
    # busNeed_df = pd.read_excel(inputName)
    pass

class CustomLemmatizer:
  _lemmatizer = WordNetLemmatizer()
  def __init__(self):
      pass

  def convertNltkTagToWordnetTag(self,nltk_tag):
    if nltk_tag.startswith('V'):
      return wordnet.VERB
    elif nltk_tag.startswith('R'):
      return wordnet.ADV
    elif nltk_tag.startswith('N'):
      return wordnet.NOUN
    elif nltk_tag.startswith('J'):
      return wordnet.ADJ
    else:          
        return None

  def lemmatize_sentence(self,sentence):
    sentence = str(sentence)
    if not sentence or sentence=='nan':
      return ""
    if sentence.isspace():
      return ""

    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  

    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], self.convertNltkTagToWordnetTag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(self._lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


class CustomDataProprocessor:
  ''' This part is for general data preprocessing part
  '''
  _cl = CustomLemmatizer()
  def __init__(self):
      pass
  
  def generateITSolModelInput(self, inputName='1.list_it_solutions_clean.csv', outputName='1.list_it_solutions_modelInput.csv'):
    '''generate model input column in IT solution which can be used as NLP model input. 
    output: 1.list_it_solutions_modelInput.csv with "Model Input" columns
    '''
    itSol_df = pd.read_csv(inputName)
    # itSol_df.loc[:,"Solution Description"] = itSol_df.loc[:,"Solution Description"].apply(lambda x: self._cl.lemmatize_sentence(x))
    # itSol_df.loc[:,"Use Case"] = itSol_df.loc[:,"Use Case"].apply(lambda x: self._cl.lemmatize_sentence(x))
    itSol_df['Model Input'] = itSol_df.loc[:,"Solution Name (Eng)"].fillna('') + ". " + itSol_df.loc[:,"Solution Description"].fillna('') + ". " + itSol_df.loc[:,"Use Case"].fillna('')

    itSol_df.to_csv(outputName, index=False)
  
  def generateBusNeedModelInput(self, inputName='2.list_business_needs.xlsx', outputName='2.list_business_needs_modelInput.csv'):
    busNeed_df = pd.read_excel(inputName)
    # busNeed_df.loc[:,'Business Needs / Challenges (Eng)'] = busNeed_df.loc[:,'Business Needs / Challenges (Eng)'].apply(lambda x: self._cl.lemmatize_sentence(x))
    # busNeed_df.loc[:,'Expected Outcomes (Eng)'] = busNeed_df.loc[:,'Expected Outcomes (Eng)'].apply(lambda x: self._cl.lemmatize_sentence(x))
    busNeed_df['Model Input'] = busNeed_df.loc[:,"Title (Eng)"] + ". " + busNeed_df.loc[:,"Business Needs / Challenges (Eng)"] + ". " + busNeed_df.loc[:,"Expected Outcomes (Eng)"]
    busNeed_df.to_csv(outputName, index=False)

if __name__ == '__main__':
  # do translation
  # ct = CustomTranslator()
  # ct.translateITSolToEn()

  # get model input
  cdp = CustomDataProprocessor()
  cdp.generateITSolModelInput()
  cdp.generateBusNeedModelInput()
