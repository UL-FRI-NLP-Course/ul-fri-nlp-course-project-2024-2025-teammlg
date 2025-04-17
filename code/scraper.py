import requests
from bs4 import BeautifulSoup
import re

class Scraper:
    def __init__(self, title, sources=["themoviedb.org", "letterboxd.com"]):
        self.title = title
        self.sources = sources

        urls = []
        for source in sources:
            if source == "letterboxd.com":

                url = "https://letterboxd.com/film/" + self.format_title(self.title) + "/reviews/by/activity"
                urls.append(url)
            #TODO add other sources, figure out how to find correct url for a queried movie, how and which subpages to visit,...

        self.urls = urls

    def format_title(self, title):
        title = str.lower(title)
        title = title.strip()
        #TODO find a better way, this is unreliable
        title = title.replace(" ", "-")
        return title

    def get_reviews(self, n):
        reviews = []
        print(self.urls[0])
        for i in range(1, n+1):
            r = requests.get(self.urls[0]+"/page/"+str(i)) #TODO generalize this
            soup = BeautifulSoup(r.content, 'html.parser')
            content = soup.find_all('div', class_='js-review-body')
            for rev in content:
                #a review can consist of multiple paragraphs - if we need this information later, we can keep it
                r = rev.find_all('p')
                total = ""

                #TODO handle js-collapsible-text ("click for more")

                for found in r:
                    total += str(found) + " "
                #strip html tags (there might be better ways to do it TODO)
                reviews.append(re.sub('<[^<]+?>', '', total))

        return reviews


    def scrape(self):
        scraped_data = []

        for url in self.urls:
            #specify how many pages of reviews you want - Letterboxd has 12 per page
            #if there aren't that many, you simply get all
            #-1 to get all (not recommended, because very popular movies can have hundreds of thousands)
            scraped_data.append(self.get_reviews(5))

        return scraped_data
    
if __name__ == "__main__":
    #usage example
    title="Challengers"
    s = Scraper(title)
    data = s.scrape()
    print(len(data[0]))
    for d in data[0]:
        print(d, "\n")


    #testing a rare movie with very few reviews
    title="Madame is Athletic"
    s = Scraper(title)
    data = s.scrape()
    print(len(data[0]))
    for d in data[0]:
        print(d, "\n")

