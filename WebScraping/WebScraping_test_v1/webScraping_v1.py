'''
Done: 
    Recursively get the sub link in the current using DFS
    Get the main content using boilerpipe Extractor
    Can extract dynamic content website
    stop when reached 2000words in the 1st time
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
import pandas as pd
import re

class RecursiveScraper:
    ''' Scrape URLs in a recursive manner.
    '''
    def __init__(self, maxWords, maxNode):
        ''' Constructor to initialize domain name and main URL.
        '''
        self.urls = set()
        self.mainContents = ""
        self.maxWords = maxWords
        self.urlQueue = []
        self.maxNode = maxNode

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
            if httpurl not in self.urls and httpsurl not in self.urls and httpsurl!="http://"+self.domain+"/index.html" and httpsurl!="https://"+self.domain+"/index.html":
                # index.html and mainurl are duplicate
                # Return URL only if it's not already in list
                return cleanurl

        return None
        
    def formaturl(self,url):
        if not re.match('(?:http|ftp|https)://', url):
            return 'http://{}'.format(url)
        return url
        
    def scrapeBFS(self, mainurl):
        self.domain = urlsplit(mainurl).netloc
        print("mainurl: ",mainurl, " self.domain: "+self.domain)
        options = webdriver.ChromeOptions()
        options.headless = True
        prefs = {'profile.default_content_setting_values': {'cookies': 2, 'images': 2, 'javascript': 2, 
                                    'plugins': 2, 'popups': 2, 'geolocation': 2, 
                                    'notifications': 2, 'auto_select_certificate': 2, 'fullscreen': 2, 
                                    'mouselock': 2, 'mixed_script': 2, 'media_stream': 2, 
                                    'media_stream_mic': 2, 'media_stream_camera': 2, 'protocol_handlers': 2, 
                                    'ppapi_broker': 2, 'automatic_downloads': 2, 'midi_sysex': 2, 
                                    'push_messaging': 2, 'ssl_cert_decisions': 2, 'metro_switch_to_desktop': 2, 
                                    'protected_media_identifier': 2, 'app_banner': 2, 'site_engagement': 2, 
                                    'durable_storage': 2}}
        options.add_experimental_option('prefs', prefs)
        options.add_argument("start-maximized")
        options.add_argument("disable-infobars")
        options.add_argument("--disable-extensions")

        browser = webdriver.Chrome('WebScraping/WebScraping_test_v1/chromedriver_win_chrome_86.exe',options=options)

        visited = {}
        num_visited_node = 0
        num_words = 0
        self.urlQueue.append(self.formaturl(mainurl))
        visited[mainurl] = True
        
        while self.urlQueue and num_visited_node<self.maxNode and num_words<self.maxWords:
            url = self.urlQueue.pop(0)
            self.urls.add(url)
            num_visited_node += 1
            
            print("browser.get url",url)
            browser.get(url)
            soup = BeautifulSoup(browser.page_source, 'lxml')
            self.mainContents += str("-------URL--------- "+url+" -------URL---------\n")
            self.num_words = self.getMainContent(browser.page_source)
            print("num_words:", self.num_words,"num_visited_node:", num_visited_node)

            for link in soup.findAll("a"):
                childurl = self.preprocess_url(url, link.get("href"))       
                if childurl and childurl not in visited and "mailto" not in childurl:
                    print("childurl", childurl)
                    self.urlQueue.append(childurl)
                    visited[childurl] = True

        browser.close()
        return None
    
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
    list_it_df = pd.read_excel("WebScraping\list_it_solutions.xls", sheet_name=0)
    website_df = list_it_df['Reference Code','Website']
    
    for index, row in website_df.iterrows():
        print("website",row['Website'])
        rscraper = RecursiveScraper(maxWords=20000, maxNode=5)
        rscraper.scrapeBFS(mainurl = row['Website'])
        with open("WebScraping\scraping_data\\"+row['Reference Code']+".txt", "w", encoding="utf-8") as f:
            f.write(rscraper.mainContents)
        print(rscraper.urls)