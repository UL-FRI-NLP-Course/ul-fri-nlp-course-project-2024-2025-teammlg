import requests
from bs4 import BeautifulSoup
import re
import json


class Scraper:
    def __init__(
        self,
        phrases,
        outfolder,
        suffix,
        sources=["tmdb", "letterboxd", "justwatch"],
        n_pages=5,
    ):
        self.phrases = phrases
        self.sources = sources
        self.data = {source: [] for source in sources}
        urls = []
        self.files = {}
        for source in sources:
            if source == "letterboxd":
                out = {}
                for movie in self.phrases["movies"]:
                    url = (
                        "https://letterboxd.com/film/"
                        + self.format_title(movie)
                        + "/reviews/by/activity"
                    )
                    urls.append(url)
                    # n_pages specifies how many pages of reviews you want - Letterboxd has 12 per page
                    # if there aren't that many, you simply get all
                    # -1 to get all (not recommended, because very popular movies can have hundreds of thousands)
                    reviews = self.get_reviews(url, n_pages)
                    self.data[source].append(reviews)
                    out[movie] = {"reviews": reviews}
                outf = outfolder + "/letterboxd_out_" + suffix + ".json"
                with open(outf, "w") as outfile:
                    json.dump(out, outfile, indent=4)
                self.files["letterboxd"] = outf
            elif source == "tmdb":
                headers = {
                    "accept": "application/json",
                    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjNzA3OTdmMTNkOGEyYjE2ODZhM2MxZTI0MzBmYmI1NCIsIm5iZiI6MTc0NDk4ODE2NC4yMjU5OTk4LCJzdWIiOiI2ODAyNjgwNDJjODVlNzk2NjM5OWJkYTYiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.prH_dcerfMwd_oxlbU6qBIaH5tqkBO3yu-z09XirBAE",
                }
                out = {}
                for movie in self.phrases["movies"]:
                    url = (
                        "https://api.themoviedb.org/3/search/movie?query="
                        + movie
                        + "&include_adult=false&language=en-US&page=1"
                    )
                    response = requests.get(url, headers=headers)
                    self.data[source].append(response.text)
                    out[movie] = {"tmdb_data": response.text}

                for person in self.phrases["people"]:
                    url = (
                        "https://api.themoviedb.org/3/search/person?query="
                        + person
                        + "&include_adult=false&language=en-US&page=1"
                    )
                    response = requests.get(url, headers=headers)
                    self.data[source].append(response.text)
                    out[person] = {"tmdb_data": response.text}

                outf = outfolder + "/tmdb_out_" + suffix + ".json"
                with open(outf, "w") as outfile:
                    json.dump(out, outfile, indent=4)
                self.files["tmdb"] = outf

            elif source == "justwatch":
                out = {}
                for movie in self.phrases["movies"]:
                    url = "https://www.justwatch.com/in/movie/" + self.format_title(
                        movie
                    )
                    urls.append(url)
                    services = self.get_services(url)
                    self.data[source].append(services)
                    out[movie] = {"services": services}

                outf = outfolder + "/justwatch_out_" + suffix + ".json"
                with open(outf, "w") as outfile:
                    json.dump(out, outfile, indent=4)
                self.files["justwatch"] = outf

            # TODO add other sources, figure out how to find correct url for a queried movie, how and which subpages to visit,...

        self.urls = urls

    def format_title(self, title):
        title = str.lower(title)
        title = title.strip()
        # TODO find a better way, this is unreliable
        title = title.replace(" ", "-")
        return title

    # there might be better ways to do it TODO
    def strip_html_tags(self, text):
        return re.sub("<[^<]+?>", "", text)

    # for justwatch
    def get_services(self, url):
        services = []
        # ugly hack, might not always work!
        headers = {
            "User-Agent": "Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
        }
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.content, "html.parser")
        content = soup.find_all("div", class_="offer__content")

        for ser in content:
            pic = ser.find("picture")
            if pic:
                txt = pic.find("img")
                if txt["alt"] not in services:
                    services.append(txt["alt"])

        return services

    # for letterboxd
    def get_reviews(self, url, n):
        reviews = []
        # TODO n=-1
        for i in range(1, n + 1):
            r = requests.get(url + "/page/" + str(i))  # TODO generalize this
            soup = BeautifulSoup(r.content, "html.parser")
            content = soup.find_all("div", class_="js-review-body")
            for rev in content:
                # a review can consist of multiple paragraphs - if we need this information later, we can keep it
                r = rev.find_all("p")
                total = ""

                # TODO handle js-collapsible-text ("click for more")

                for found in r:
                    total += str(found)

                reviews.append(self.strip_html_tags(total))

        return reviews


if __name__ == "__main__":
    # usage example
    # phrases = {"movies": ["challengers"], "people": []}
    # s = Scraper(phrases, "data/scraped_data", "", sources=["tmdb"])
    # data = s.data

    # testing a rare movie with very few reviews
    # phrases = {"movies": ["Madame is Athletic", "the godfather"], "people": ["bruce willis", "king kong"]}
    # s = Scraper(phrases, "data/scraped_data", "")
    # data = s.data

    phrases = {"movies": ["challengers", "the godfather"], "people": []}
    s = Scraper(phrases, "data/scraped_data", "", sources=["justwatch"])
