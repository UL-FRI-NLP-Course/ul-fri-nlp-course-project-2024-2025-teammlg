import enum
import json
import logging
import os
import re
from typing import Dict, List, TypedDict

from bs4 import BeautifulSoup
import requests

from extraction import ExtractedData


class ScraperSource(enum.StrEnum):
    TMDB = "tmdb"
    Letterboxd = "letterboxd"
    Wikipedia = "wiki"
    JustWatch = "justwatch"


class Scraper:
    def __init__(self, sources: List[ScraperSource], n_pages: int = 5, logging_level: int = logging.INFO):
        self.headers = {
            "accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjNzA3OTdmMTNkOGEyYjE2ODZhM2MxZTI0MzBmYmI1NCIsIm5iZiI6MTc0NDk4ODE2NC4yMjU5OTk4LCJzdWIiOiI2ODAyNjgwNDJjODVlNzk2NjM5OWJkYTYiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.prH_dcerfMwd_oxlbU6qBIaH5tqkBO3yu-z09XirBAE"
        }
        self.genres = self.get_all_genres()
        self.sources = sources
        self.files = {}
        self.n_pages = n_pages

        name = "Scraper"
        self.output_directory = "scraping_results"
        out_path = os.path.join(self.output_directory, f"{name}.log")

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging_level)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.addHandler(logging.FileHandler(out_path))
    
    def scrape(self, input_data: ExtractedData) -> Dict[ScraperSource, str]:
        results = {}
        
        for source in self.sources:
            if source == ScraperSource.Letterboxd:
                letterboxd_file_store = {}
                logging.info("Scraping letterboxd...")
                for movie in input_data["movies"]:
                    letterboxd_reviews = self.scrape_letterboxd(movie, self.n_pages)
                    letterboxd_file_store[movie] = letterboxd_reviews
                results[ScraperSource.Letterboxd.value] = letterboxd_file_store

                store_filepath = os.path(self.output_directory, "letterboxd.json")
                with open(store_filepath, "w", encoding="utf-8") as file:
                    json.dump(letterboxd_file_store, file, ensure_ascii=False)
            
            elif source == ScraperSource.TMDB:
                tmdb_file_store = {}
                logging.info("Scraping TMDB...")
                for movie in input_data["movies"]:
                    tmdb_result = self.scrape_tmdb_movie_info(movie)
                    tmdb_file_store[movie] = tmdb_result
                results[ScraperSource.TMDB.value] = tmdb_file_store
                
                store_filepath = os.path(self.output_directory, "tmdb.json")
                with open(store_filepath, "w", encoding="utf-8") as file:
                    json.dump(tmdb_file_store, file, ensure_ascii=False)
            
            elif source == ScraperSource.JustWatch:
                justwatch_file_store = {}
                logging.info("Scraping JustWatch...")
                for movie in input_data["movies"]:
                    justwatch_result = self.scrape_justwatch(movie)
                    justwatch_file_store[movie] = justwatch_result
                results[ScraperSource.JustWatch.value] = justwatch_file_store

                store_filepath = os.path(self.output_directory, "justwatch.json")
                with open(store_filepath, "w", encoding="utf-8") as file:
                    json.dump(justwatch_file_store, file, ensure_ascii=False)

            elif source == ScraperSource.Wikipedia:
                wikipedia_file_store = {}
                logging.info("Scraping Wikipedia...")
                for movie in input_data["movies"]:
                    wikipedia_result = self.scrape_wikipedia(movie)
                    wikipedia_file_store[movie] = wikipedia_result
                results[ScraperSource.Wikipedia.value] = wikipedia_file_store

                store_filepath = os.path(self.output_directory, "wikipedia.json")
                with open(store_filepath, "w", encoding="utf-8") as file:
                    json.dump(wikipedia_file_store, file, ensure_ascii=False)
        return results

    def scrape_letterboxd(self, movie: str, n_pages: int) -> List[str]:
        possible = self.get_possible_urls(movie)
        reviews: List[str] = []
        for p in possible:
            url = ("https://letterboxd.com/film/" + p + "/reviews/by/activity")
            logging.info(f"Scraping {url}")
            # n_pages specifies how many pages of reviews you want - Letterboxd has 12 per page
            # if there aren't that many, you simply get all
            revs = self.get_letterboxd_reviews(url, n_pages)
            if len(revs) > 0:
                reviews.append(revs)
        flattened_reviews = [review for review_batch in reviews for review in review_batch]
        return flattened_reviews
    
    def get_letterboxd_reviews(url: str, n: int) -> List[str]:
        def strip_html_tags(text: str) -> str:
            return re.sub("<[^<]+?>", "", text)

        reviews: List[str] = []
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

    def scrape_tmdb_movie_info(self, movie: str) -> Dict[str, str]:
        # TODO
        pass

    def scrape_justwatch(self, movie: str) -> Dict[str, str]:
        # TODO
        pass

    def scrape_wikipedia(self, movie: str) -> Dict[str, str]:
        # TODO
        pass

    def get_all_genres(self, include_tv_genres: bool = False) -> Dict[str, str]:
        url = "https://api.themoviedb.org/3/genre/movie/list?language=en"
        response = requests.get(url, headers=self.headers)
        genres = {}
        for g in response.json()["genres"]:
            genres[g["id"]] = g["name"]

        if include_tv_genres:
            url = "https://api.themoviedb.org/3/genre/tv/list?language=en"
            response = requests.get(url, headers=self.headers)
            for g in response.json()["genres"]:
                genres[g["id"]] = g["name"]
        return genres 
    
    def get_possible_urls(self, movie_name: str, expected_number_of_movies_from_same_year: int = 5) -> List[str]:
        def format_title(title: str) -> str:
            title = str.lower(title)
            title = title.strip()
            title = title.replace(" ", "-")
            title = title.replace("\'", "")
            title = title.replace("’", "")
            return title

        # first extract the year from tmdb
        url = ("https://api.themoviedb.org/3/search/movie?query=" + movie_name + "&include_adult=false&language=en-US&page=1")
        response = requests.get(url, headers=self.headers)
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