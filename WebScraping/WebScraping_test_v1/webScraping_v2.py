'''
Done: 
    Recursively get the sub link in the current using DFS
    Get the main content using boilerpipe Extractor
Not Done:
    Change to BFS to search for sub link and extract
    version control using git
     
'''

from boilerpipe.extract import Extractor
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlsplit, SplitResult
import requests
import nltk
from selenium import webdriver

class RecursiveScraper:
    ''' Scrape URLs in a recursive manner.
    '''
    def __init__(self, url, maxWords):
        ''' Constructor to initialize domain name and main URL.
        '''
        self.domain = urlsplit(url).netloc
        self.mainurl = url
        self.urls = set()
        self.mainContents = ""
        self.maxWords = maxWords
        self.countWords = 0

    def preprocess_url(self, referrer, url):
        ''' Clean and filter URLs before scraping.
        '''
        if not url:
            return None

        fields = urlsplit(urljoin(referrer, url))._asdict() # convert to absolute URLs and split
        fields['path'] = re.sub(r'/$', '', fields['path']) # remove trailing /
        fields['fragment'] = '' # remove targets within a page
        fields = SplitResult(**fields)
        if fields.netloc == self.domain:
            # Scrape pages of current domain only
            if fields.scheme == 'http':
                httpurl = cleanurl = fields.geturl()
                httpsurl = httpurl.replace('http:', 'https:', 1)
            else:
                httpsurl = cleanurl = fields.geturl()
                httpurl = httpsurl.replace('https:', 'http:', 1)
            if httpurl not in self.urls and httpsurl not in self.urls:
                # Return URL only if it's not already in list
                return cleanurl

        return None

    def scrape(self, url=None):
        ''' Scrape the URL and its outward links in a depth-first order.
            If URL argument is None, start from main page.
        '''
        if url is None:
            url = self.mainurl

        if self.countWords>self.maxWords:
            return
        
        print("Scraping {:s} ...".format(url))
        self.urls.add(url)

        opts = webdriver.ChromeOptions()
        opts.headless = True
        browser = webdriver.Chrome('./chromedriver',options=opts)
        browser.get(url)
        soup = BeautifulSoup(browser.page_source, 'lxml')

        self.countWords = self.getMainContent(browser.page_source)
        print("size:", self.countWords)
        browser.close()
        
        # print("a", [ link.get("href") for link in soup.findAll("a")])
        for link in soup.findAll("a"):
            childurl = self.preprocess_url(url, link.get("href"))
            if childurl:
                self.scrape(childurl)

    
    def getMainContent(self,html):
        if html is None:
            return self.tokenizeText(self.mainContents)
        try:
            extractor = Extractor(extractor='DefaultExtractor', html=html, kMin=20)
        except:
            print("extract error")
            return self.tokenizeText(self.mainContents)
        extracted_text = extractor.getText()
        self.mainContents += str(extracted_text)
        return self.tokenizeText(self.mainContents)

    
    def tokenizeText(self, text):
        sent_text = nltk.sent_tokenize(text) # this gives us a list of sentences
        # now loop over each sentence and tokenize it separately
        texts = []
        for sentence in sent_text:
            tokenized_text = nltk.word_tokenize(sentence)
            # tagged = nltk.pos_tag(tokenized_text)
            texts += [x for x in tokenized_text if len(x) > 1]
        return len(texts)


if __name__ == '__main__':
    ITurl = 'http://www.sap.com'
    rscraper = RecursiveScraper(url=ITurl, maxWords=2000)
    rscraper.scrape()
    with open("output.txt", "w") as f:
        f.write(rscraper.mainContents)
    print(rscraper.urls)
