
import re 
import string 
import nltk 
import spacy 
import pandas as pd 
import numpy as np 
import math 
from tqdm import tqdm 

from spacy.matcher import Matcher 
from spacy.tokens import Span 
from spacy import displacy 
import pandas as pd
data = pd.ExcelFile("*File Name*")

nlp = spacy.load('en_core_web_sm')
output = open("WebScraping\Spacy\output.txt", encoding="utf-8").read()
outputDoc = nlp(output)
# for num, sentence in enumerate(doc.sents):
#     print(f'{num}: {sentence}')
# for token in doc:
#     print(f'{token.text}')
# [token.text for token in doc]

# text = "SAP was founded in 1972 in Walldorf, Germany and now has offices around the world."
# sampledoc = nlp(text)
# for tok in sampledoc:
#     print(tok.text, "-->", tok.dep_, "-->", tok.pos_)

pattern = [{'POS': 'PROPN'},
           {'POS': 'AUX'},
           {'LOWER': 'founded'},
           {'LOWER': 'in'},
           {'POS': 'NUM'}]

# matcher = Matcher(nlp.vocab)
# matcher.add("matching_1", None, pattern)

# matches = matcher(ouputDoc)
# span = ouputDoc[matches[0][1]:matches[0][2]]
# print(span.text)

def subtree_matcher(doc):
  subjpass = 0

  for i,tok in enumerate(doc):
    # find dependency tag that contains the text "subjpass"    
    if tok.dep_.find("subjpass") == True:
      subjpass = 1

  x = ''
  y = ''

  # if subjpass == 1 then sentence is passive
  if subjpass == 1:
    for i,tok in enumerate(doc):
      if tok.dep_.find("subjpass") == True:
        y = tok.text

      if tok.dep_.endswith("obj") == True:
        x = tok.text
  
  # if subjpass == 0 then sentence is not passive
  else:
    for i,tok in enumerate(doc):
      if tok.dep_.endswith("subj") == True:
        x = tok.text

      if tok.dep_.endswith("obj") == True:
        y = tok.text

  return x,y

# print(subtree_matcher(sampledoc))
for ent in outputDoc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)