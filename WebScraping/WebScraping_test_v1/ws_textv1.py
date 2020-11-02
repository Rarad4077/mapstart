from boilerpipe.extract import Extractor
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlsplit, SplitResult
import requests
import nltk
from selenium import webdriver

url = 'http://www.bithkex.com/bitex_v2/about.html'
# response = requests.get(url)
# soup = BeautifulSoup(response.content, 'lxml')

# browser = webdriver.Safari()
# browser.get(url)

opts = webdriver.ChromeOptions()
opts.headless = True
browser = webdriver.Chrome('./chromedriver',options=opts)
browser.get(url)

soup = BeautifulSoup(browser.page_source, 'lxml')

with open("outputText.txt", "w") as f:
    extractor = Extractor(extractor='DefaultExtractor', html=browser.page_source, kMin=20)
    f.write(str(extractor.getText()))

browser.close()