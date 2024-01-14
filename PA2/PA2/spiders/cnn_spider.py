import scrapy
import hashlib
#import requests
from bs4 import BeautifulSoup
from scrapy.linkextractors import LinkExtractor
import re
from collections import Counter
from nltk.corpus import stopwords

#json_file=

class CNNSpider(scrapy.Spider):
    name = "cnn"
    allowed_domains= ["cnn.com"]

    start_urls = [
        "https://www.cnn.com/articles",
        "https://www.cnn.com/us/energy-and-environment",
        "https://www.cnn.com/us",
        "https://www.cnn.com/world/",
        "https://www.cnn.com/sport",
        "https://www.cnn.com/style",
        "https://www.cnn.com/travel",
        "https://www.cnn.com/politics",
        "https://www.cnn.com/business",
        "https://www.cnn.com/health",
        "https://www.cnn.com/business/tech",
        "https://www.cnn.com/entertainment",
        "https://www.cnn.com/opinions",
        "https://www.cnn.com/markets",
        "https://www.cnn.com/weather"

    ]



    def parse(self, response):
        entry= dict.fromkeys(['pageid', 'url', 'title', 'article'])
        
        href_match="\W*\/2023\S*"
        encoding_key=response.url
        
        entry['pageid']= hashlib.md5(encoding_key.encode('utf-8')).hexdigest()
        entry['url']=response.url
        entry['title']= response.css("title::text").get()
        entry['article']= response.css("p.paragraph.inline-placeholder::text").getall()
        yield entry

        for href in response.css("a::attr(href)").re(href_match):
            yield response.follow(href,callback=self.parse)
                
        





    
            
            