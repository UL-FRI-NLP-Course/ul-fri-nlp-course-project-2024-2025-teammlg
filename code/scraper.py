from typing import Any, Dict, List
import numpy
import requests
from bs4 import BeautifulSoup
import re
import json
import wikipedia
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings('ignore')

with open('./data/stopwords-en.txt', "r", encoding="utf8") as f:
    stop_words = f.readlines()
    stop_words = [word.strip() for word in stop_words]

class Scraper:
    def __init__(self, phrases, outfolder, suffix, sources=["tmdb", "letterboxd", "justwatch", "wiki"], n_pages=5):
        self.phrases = phrases
        self.sources = sources
        self.data = {source: [] for source in sources}
        urls = []
        self.files = {}

        self.headers = {
            "accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjNzA3OTdmMTNkOGEyYjE2ODZhM2MxZTI0MzBmYmI1NCIsIm5iZiI6MTc0NDk4ODE2NC4yMjU5OTk4LCJzdWIiOiI2ODAyNjgwNDJjODVlNzk2NjM5OWJkYTYiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.prH_dcerfMwd_oxlbU6qBIaH5tqkBO3yu-z09XirBAE"
        }

        GENRES = self.get_all_genres(self.headers)

        for source in sources:
            if source == "letterboxd":
                print("Started scraping letterboxd")
                out = {}
                for movie in self.phrases["movies"]:
                    possible = self.get_possible_urls(movie, self.headers)
                    reviews = []
                    for p in possible:
                        url = ("https://letterboxd.com/film/" + p + "/reviews/by/activity")
                        urls.append(url)
                        # n_pages specifies how many pages of reviews you want - Letterboxd has 12 per page
                        # if there aren't that many, you simply get all
                        revs = self.get_letterboxd_reviews(url, n_pages)
                        if len(revs) > 0:
                            reviews.append(revs)
                    self.data[source].append(reviews)
                    out[movie] = {"reviews": reviews}
                outf = outfolder + "/letterboxd_out_" + suffix + ".json"
                with open(outf, "w", encoding="utf8") as outfile:
                    json.dump(out, outfile, indent=4, ensure_ascii=False)
                self.files["letterboxd"] = outf
                print("Finished scraping letterboxd")
            elif source == "tmdb":
                print("Started scraping tmdb")
                out = {}
                for movie in self.phrases["movies"]:
                    url = ("https://api.themoviedb.org/3/search/movie?query=" + movie + "&include_adult=false&language=en-US&page=1")
                    response = requests.get(url, headers=self.headers)
                    self.data[source].append(response.text)

                    if "results" in response.json().keys():
                        res = response.json()["results"]
                        if len(res) > 0:
                            id = res[0]["id"]
                            cast = self.get_cast_list(id, self.headers)
                            similar = self.get_similar_movies(id, self.headers)
                        else:
                            cast = ""
                            similar = ""
                    else:
                        cast = ""
                        similar = ""

                    # Take resulting JSON and select the most appropriate title
                    movie_json = self.select_most_appropriate_title(response.json(), self.phrases["key"])

                    # TODO: Put this in front of selection of title
                    # The result contains only genre IDs, so we convert them to genres
                    if "genre_ids" in movie_json.keys():
                        genre_ids = movie_json.get("genre_ids", [])
                        genres = [GENRES.get(g, "") for g in genre_ids]
                        movie_json["genres"] = genres
                    #else:
                    #    movie_json["genres"] = None

                    # We delete some unnecessary fields to recude the prompt size
                    if movie_json.keys():
                        del movie_json["genre_ids"]
                        del movie_json["id"]
                        del movie_json["backdrop_path"]
                        del movie_json["poster_path"]

                    out[movie] = {"info": movie_json, "similar": similar, "cast": cast}

                for person in self.phrases["people"]:
                    url = ("https://api.themoviedb.org/3/search/person?query="+ person+ "&include_adult=false&language=en-US&page=1")
                    response = requests.get(url, headers=self.headers)
                    
                    credits={}
                    # currently we get credits for the most popular person with the name, could be benefitial to get all?
                    if "results" in response.json().keys():
                        max_popularity = 0
                        max_id = -1
                        for result in response.json()["results"]:
                            if "id" in result.keys() and "popularity" in result.keys() and result["popularity"] > max_popularity:
                                # if you want raw data add raw=True, there's a lot of junk in there though
                                max_id = result["id"]
                                max_popularity = int(result["popularity"])
                        
                        credits[result["id"]] = self.get_credits_for_person(max_id, self.headers)

                    self.data[source].append(response.text)
                    out[person] = {"credits":credits, "info": response.json()}

                outf = outfolder + "/tmdb_out_" + suffix + ".json"
                with open(outf, "w", encoding="utf8") as outfile:
                    json.dump(out, outfile, indent=4, ensure_ascii=False)
                self.files["tmdb"] = outf
                print("Finished scraping tmdb")

            elif source == "justwatch":
                print("Started scraping justwatch")
                out = {}
                for movie in self.phrases["movies"]:
                    possible = self.get_possible_urls(movie, self.headers)
                    services = []
                    for p in possible:
                        url = "https://www.justwatch.com/in/movie/" + p
                        urls.append(url)

                        serv = self.get_streaming_services(url)
                        if len(serv) > 0:
                            services.append(serv)
                    self.data[source].append(services)
                    out[movie] = {"services": services}

                outf = outfolder + "/justwatch_out_" + suffix + ".json"
                with open(outf, "w", encoding="utf8") as outfile:
                    json.dump(out, outfile, indent=4, ensure_ascii=False)
                self.files["justwatch"] = outf
                print("Finished scraping justwatch")

            elif source == "wiki":
                print("Started scraping wiki")
                out = {}
                for movie in self.phrases["movies"]:
                    wiki = self.get_wikipedia_data(movie, self.headers)
                    out[movie] = wiki
                
                outf = outfolder + "/wiki_out_" + suffix + ".json"
                with open(outf, "w", encoding="utf8") as outfile:
                    json.dump(out, outfile, indent=4, ensure_ascii=False)
                self.files["wiki"] = outf
                print("Finished scraping wiki")

        self.urls = urls

    @staticmethod
    def get_wikipedia_data(movie: str, headers: str) -> Dict[str, str]:
        movie = movie.strip()
        #movie = movie.replace("\'", "%27")
        #movie = movie.replace("\'", "")
        out = {}
        
        url = ("https://api.themoviedb.org/3/search/movie?query=" + movie + "&include_adult=false&language=en-US&page=1")
        response = requests.get(url, headers=headers)
        results = response.json()["results"]
        if len(results) == 0:
            return {}
        
        # TODO pick most popular?
        year = results[0]["release_date"][:4]

        possible_titles = [movie, movie + " (film)", movie + " (" + year + " film)"]
        possible_sections = ["Plot", "Synopsis", "Summary", "Plot synopsis", "Plot summary", "Story"]

        for title in possible_titles:
            success = False
            plot = ""
            try:
                wiki = wikipedia.WikipediaPage(title = title)
                summary = wiki.summary

                for section in possible_sections:
                    try:
                        attempt = wiki.section(section)
                        
                        if attempt:
                            plot += attempt
                            success = True
                            break # one found thing should be enough
                    except:
                        continue
            except:
                continue

            # if we haven't found any variant of plot, we're probably not on the correct movie page, so we should discard the summary as well
            if not success:
                out[movie] = {"summary": "", "plot": ""}
            else:
                out[movie] = {"summary": summary, "plot": plot}

        return out

    @staticmethod
    def get_similar_movies(movie_id: str, headers: str, raw: bool = False) -> List[str]:
        url = "https://api.themoviedb.org/3/movie/"+str(movie_id)+"/similar?language=en-US&page=1"
        response = requests.get(url, headers=headers)

        if raw:
            return response.text

        data = response.json()["results"]
        movies = []
        for d in data:
            movies.append(d["title"])

        return movies

    @staticmethod
    def get_possible_urls(movie_name: str, headers: str, expected_number_of_movies_from_same_year: int = 5) -> List[str]:
        def format_title(title: str) -> str:
            title = str.lower(title)
            title = title.strip()
            title = title.replace(" ", "-")
            title = title.replace("\'", "")
            title = title.replace("’", "")
            return title

        # first extract the year from tmdb
        url = ("https://api.themoviedb.org/3/search/movie?query=" + movie_name + "&include_adult=false&language=en-US&page=1")
        response = requests.get(url, headers=headers)
        results = response.json()["results"]
        if len(results) == 0:
            return [movie_name]
        base = format_title(results[0]["title"]) # hopefully this handles potential misspellings
        movies = [base]
        
        # TODO pick most popular?
        year = results[0]["release_date"][:4]

        movies.append(base + "-" + year)

        # ultra rare cases of multiple movies in the same year with the same title
        for i in range(expected_number_of_movies_from_same_year):
            movies.append(base + "-" + year + "-" + str(i+1))

        return movies

    @staticmethod
    def get_all_genres(headers: str, include_tv_genres: bool = False) -> Dict[str, str]:
        url = "https://api.themoviedb.org/3/genre/movie/list?language=en"
        response = requests.get(url, headers=headers)
        genres = {}
        for g in response.json()["genres"]:
            genres[g["id"]] = g["name"]

        if include_tv_genres:
            url = "https://api.themoviedb.org/3/genre/tv/list?language=en"
            response = requests.get(url, headers=headers)
            for g in response.json()["genres"]:
                genres[g["id"]] = g["name"]
        return genres 

    @staticmethod
    def get_credits_for_person(person_id: str, headers: str, raw: bool = False) -> Dict[str, str]:
        url = "https://api.themoviedb.org/3/person/"+str(person_id)+"/movie_credits?language=en-US"
        response = requests.get(url, headers=headers)
        if raw:
            return str(response.text)
        
        out = {"cast":[], "crew":[]}
        # cleanup
        response = response.json()
        if "cast" in response:
            for r in response["cast"]:
                cast = r
                if cast:
                    del cast["adult"]
                    del cast["backdrop_path"]
                    del cast["original_language"]
                    del cast["poster_path"]
                    del cast["video"]
                    del cast["vote_average"]
                    del cast["vote_count"]
                    del cast["credit_id"]
                    del cast["order"]
                out["cast"].append(cast)
        if "crew" in response:
            for r in response["crew"]:
                crew = r
                if crew:
                    del crew["adult"]
                    del crew["backdrop_path"]
                    del crew["original_language"]
                    del crew["poster_path"]
                    del crew["video"]
                    del crew["vote_average"]
                    del crew["vote_count"]
                    del crew["credit_id"]
                out["crew"].append(crew)
        return out

    @staticmethod
    def get_cast_list(movie_id: str, headers: str, raw: bool = False) -> Dict[str, str]:
        url = "https://api.themoviedb.org/3/movie/"+str(movie_id)+"/credits?language=en-US"
        response = requests.get(url, headers=headers)
        
        if raw:
            return str(response.text)
        
        out = {"cast":[], "crew":[]}
        # cleanup
        response = response.json()
        if "cast" in response:
            for r in response["cast"]:
                cast = r
                if cast:
                    del cast["adult"]
                    del cast["id"]
                    del cast["profile_path"]
                    del cast["cast_id"]
                    del cast["credit_id"]
                    del cast["order"]
                out["cast"].append(cast)
        if "crew" in response:
            for r in response["crew"]:
                crew = r
                if crew:
                    del crew["adult"]
                    del crew["id"]
                    del crew["profile_path"]
                    del crew["credit_id"]
                out["crew"].append(crew)
        return out

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
        if len(documents) == 0:
            return {}
        phrases = " ".join(key_phrases)

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

    @staticmethod
    def get_streaming_services(url: str) -> List[str]:
        services = []
        # ugly hack, might not always work!
        headers = {"User-Agent": "Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"}
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

    @staticmethod
    def get_letterboxd_reviews(url: str, n: int) -> List[str]:
        def strip_html_tags(text: str) -> str:
            return re.sub("<[^<]+?>", "", text)

        reviews = []
        for i in range(1, n + 1):
            # ugly hack, but it's the best we have
            headers = {"User-Agent": "Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"}
            r = requests.get(url + "/page/" + str(i), headers=headers) 
            soup = BeautifulSoup(r.content, "html.parser")
            #content = soup.find_all("div", class_="js-review-body")
            content = soup.find_all("div", class_="js-review")
            for rev in content:
                r = rev.find_all("p")
                total = ""

                for found in r:
                    total += str(found)

                total = total.replace("This review may contain spoilers. I can handle the truth.", "")
                total = strip_html_tags(total).strip()

                reviews.append(total)

        return reviews

if __name__ == "__main__":
    # usage example
    phrases = {"movies": ["inheritance", "challengers", "the godfather", "the hands of orlac", "i'm still here"], "people": ["chuck jones"], "key": ""}
    #s = Scraper(phrases, "data/scraped_data", "", sources=["wiki"])
    s = Scraper(phrases, "data/scraped_data", "", sources=["tmdb"])
