import requests
from bs4 import BeautifulSoup

class Scraper:
    def __init__(self, title, sources=["themoviedb.org", "letterboxd.com"]):
        self.title = str.lower(title)
        self.sources = sources
    
        urls = []
        url = ""
        for source in sources:
            if source == "letterboxd.com":
                url = "https://letterboxd.com/film/" + self.title
                urls.append(url)
            #TODO add other sources, figure out how to find correct url for a queried movie, how and which subpages to visit,...

        self.urls = urls

    def scrape(self):
        scraped_data = []

        for url in self.urls:
            r = requests.get(url)
            soup = BeautifulSoup(r.content, 'html.parser')
            scraped_data.append(soup)

        return scraped_data
    
#usage example
title="Challengers"
s = Scraper(title)
data = s.scrape()
print(data)
