from typing import Any, Dict, List, Optional
import nltk
import numpy
import requests
from bs4 import BeautifulSoup
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer


# TODO: Down the road, use https://api.themoviedb.org/3/genre/movie/list to get list of valid genres
GENRES = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
}


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
        # TODO: We need a way to ensure that the movie/series on Letterboxd is the same as on TMDB and JustWatch
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
                with open(outf, "w", encoding="utf8") as outfile:
                    json.dump(out, outfile, indent=4, ensure_ascii=False)
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

                    # Take resulting JSON and select the most appropriate title
                    movie_json = self.select_most_appropriate_title(
                        response.json(), self.phrases["key"]
                    )

                    # TODO: Put this in front of selection of title
                    # The result contains only genre IDs, so we convert them to genres
                    genre_ids = movie_json.get("genre_ids", [])
                    genres = [GENRES.get(g, "") for g in genre_ids]
                    movie_json["genres"] = genres

                    # We delete some unnecessary fields to recude the prompt size
                    del movie_json["genre_ids"]
                    del movie_json["id"]
                    del movie_json["backdrop_path"]
                    del movie_json["poster_path"]

                    out[movie] = {"tmdb_data": movie_json}

                for person in self.phrases["people"]:
                    url = (
                        "https://api.themoviedb.org/3/search/person?query="
                        + person
                        + "&include_adult=false&language=en-US&page=1"
                    )
                    response = requests.get(url, headers=headers)
                    self.data[source].append(response.text)
                    out[person] = {"tmdb_data": response.json()}

                outf = outfolder + "/tmdb_out_" + suffix + ".json"
                with open(outf, "w", encoding="utf8") as outfile:
                    json.dump(out, outfile, indent=4, ensure_ascii=False)
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
                with open(outf, "w", encoding="utf8") as outfile:
                    json.dump(out, outfile, indent=4, ensure_ascii=False)
                self.files["justwatch"] = outf

            # TODO add other sources, figure out how to find correct url for a queried movie, how and which subpages to visit,...

        self.urls = urls

    @staticmethod
    def json_to_plain_text(j, clear_all: bool = False) -> str:
        j = str(j)
        j = j.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
        if clear_all:
            j = j.replace('"', " ")
            j = j.replace("'", " ")
        return j

    @staticmethod
    def select_most_appropriate_title(data: Dict[str, Any], key_phrases: List[str]):
        """A really primitive selection process. Just check how many times some key phrase
        is found in the result and select the one with the most occurances."""
        documents = data["results"]
        phrases = " ".join(key_phrases)

        stop_words = list(nltk.corpus.stopwords.words("english"))
        vectorizer = TfidfVectorizer(stop_words=stop_words)
        text_documents = [Scraper.json_to_plain_text(doc) for doc in documents]
        tfidf = vectorizer.fit_transform([phrases, *text_documents])

        phrases_tfidf = tfidf[:, 0]

        max_score = 0
        best_document = {}
        for i, document in enumerate(documents):
            popularity = document.get("popularity", 0.0)
            document_tfidf = tfidf[:, i + 1]
            comparison = numpy.sum(phrases_tfidf * document_tfidf.T)
            comparison = 0.5 * comparison + popularity * 0.5
            if comparison > max_score:
                best_document = document
                max_score = comparison
        return best_document

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
