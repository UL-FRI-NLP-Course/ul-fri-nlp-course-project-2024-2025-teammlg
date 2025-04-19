import requests
from bs4 import BeautifulSoup
import re
import json

class Scraper:
    def __init__(self, title, sources=["themoviedb.org", "letterboxd.com"], n_pages=5):
        self.title = title
        self.sources = sources
        self.data = {source:[] for source in sources}
        urls = []
        for source in sources:
            if source == "letterboxd.com":
                url = "https://letterboxd.com/film/" + self.format_title(self.title) + "/reviews/by/activity"
                urls.append(url)
                #n_pages specifies how many pages of reviews you want - Letterboxd has 12 per page
                #if there aren't that many, you simply get all
                #-1 to get all (not recommended, because very popular movies can have hundreds of thousands)
                reviews = self.get_reviews(url, n_pages)
                self.data[source] = reviews
                out = {'reviews': reviews} 
                with open("data/scraped_data/letterboxd_out.json", "w") as outfile:
                    json.dump(out, outfile, indent=4)
            elif source == "themoviedb.org":
                url = "https://api.themoviedb.org/3/search/movie?query="+title+"&include_adult=false&language=en-US&page=1"
                headers = {
                    "accept": "application/json",
                    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjNzA3OTdmMTNkOGEyYjE2ODZhM2MxZTI0MzBmYmI1NCIsIm5iZiI6MTc0NDk4ODE2NC4yMjU5OTk4LCJzdWIiOiI2ODAyNjgwNDJjODVlNzk2NjM5OWJkYTYiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.prH_dcerfMwd_oxlbU6qBIaH5tqkBO3yu-z09XirBAE"
                }
                response = requests.get(url, headers=headers)
                self.data[source] = response.text
                with open("data/scraped_data/tmdb_out.json", "w") as outfile:
                    json.dump(response.text, outfile, indent=4)
            #TODO add other sources, figure out how to find correct url for a queried movie, how and which subpages to visit,...

        self.urls = urls

    def format_title(self, title):
        title = str.lower(title)
        title = title.strip()
        #TODO find a better way, this is unreliable
        title = title.replace(" ", "-")
        return title

    #for letterboxd
    def get_reviews(self, url, n):
        reviews = []
        #TODO n=-1
        for i in range(1, n+1):
            r = requests.get(url+"/page/"+str(i)) #TODO generalize this
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
    
if __name__ == "__main__":
    #usage example
    title="Challengers"
    s = Scraper(title, sources=["themoviedb.org"])
    data = s.data

    #testing a rare movie with very few reviews
    title="Madame is Athletic"
    s = Scraper(title)
    data = s.data
    #print(len(data[0]))
    #for d in data[0]:
    #    print(d, "\n")

