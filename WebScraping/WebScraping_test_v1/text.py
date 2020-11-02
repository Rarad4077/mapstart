from boilerpipe.extract import Extractor



ITurl = 'https://www.sap.com/'
extractor = Extractor(extractor='DefaultExtractor', url=ITurl, kMin=20)
extracted_text = extractor.getText()
print(extracted_text)



