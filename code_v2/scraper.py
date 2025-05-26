import re
from typing import Dict, List
from bs4 import BeautifulSoup
import requests
import wikipedia


HEADERS = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjNzA3OTdmMTNkOGEyYjE2ODZhM2MxZTI0MzBmYmI1NCIsIm5iZiI6MTc0NDk4ODE2NC4yMjU5OTk4LCJzdWIiOiI2ODAyNjgwNDJjODVlNzk2NjM5OWJkYTYiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.prH_dcerfMwd_oxlbU6qBIaH5tqkBO3yu-z09XirBAE"
}

GENRES = {}

def get_all_genres(include_tv_genres: bool = False):
    url = "https://api.themoviedb.org/3/genre/movie/list?language=en"
    response = requests.get(url, headers=HEADERS)
    for g in response.json()["genres"]:
        GENRES[g["id"]] = g["name"]

    if include_tv_genres:
        url = "https://api.themoviedb.org/3/genre/tv/list?language=en"
        response = requests.get(url, headers=HEADERS)
        for g in response.json()["genres"]:
            GENRES[g["id"]] = g["name"]

get_all_genres()


def get_random_movies_based_on_language(language: str) -> List:
    """
    Finds some random movies based on spoken language.

    Args:
        language (str): The language in the movie. American English is 'en-US', etc.

    Returns:
        List: A list of movie titles, their release dates, and premises
    """
    url = f"https://api.themoviedb.org/3/discover/movie?language={language}"
    response = requests.get(url, headers=HEADERS)
    results = response.json()["results"]
    return [{
        "title": result["title"],
        "release_date": result["release_date"],
        "genres": [GENRES[genre_id] for genre_id in result["genres"]]
    } for result in results]



def get_movie_release_date(title: str) -> Dict[str, str]:
    """
    Get the release date for a movie.

    Args:
        title: The title of the movie.

    Returns:
        The release dates of all movies with that or similar title, in format 'YYYY-mm-dd'.
    """

    url = ("https://api.themoviedb.org/3/search/movie?query=" + title + "&include_adult=false&language=en-US&page=1")
    response = requests.get(url, headers=HEADERS)
    results = response.json()["results"]
    release_date = results[0]["release_date"]
    results = {
        title: release_date
    }
    return results

def get_movie_genres(title: str) -> Dict[str, List[str]]:
    """
    Get the list of genres of a movie.

    Args:
        title: The title of the movie.

    Returns:
        The genres of all movies with that or similar title, as a list of strings.
    """

    url = ("https://api.themoviedb.org/3/search/movie?query=" + title + "&include_adult=false&language=en-US&page=1")
    response = requests.get(url, headers=HEADERS)
    results = response.json()["results"]
    if not results:
        return {title: []}
    genre_ids = results[0]["genre_ids"]
    genres = [GENRES[g] for g in genre_ids]            

    return {
        title: genres
    }

def get_movie_cast_list(title: str) -> Dict[str, List[Dict]]:
    """
    Get the cast list for a movie and additional information about each cast member.

    Args:
        title: The title of the movie.

    Returns:
        The cast lists of all movies with that or similar title, as a list of dicts.
    """

    url = ("https://api.themoviedb.org/3/search/movie?query=" + title + "&include_adult=false&language=en-US&page=1")
    response = requests.get(url, headers=HEADERS)
    results = response.json()["results"]
    if not results:
        return {"cast": [{}]}
    movie_id = results[0]["id"]

    url = "https://api.themoviedb.org/3/movie/"+str(movie_id)+"/credits?language=en-US"
    response = requests.get(url, headers=HEADERS)
    
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

def get_movie_streaming_services(title: str) -> Dict[str, List[str]]:
    """
    Get the streaming services for a movie.

    Args:
        title: The title of the movie.

    Returns:
        The streaming services of all movies with that or similar title, as a list of strings.
    """
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
        
        year = results[0]["release_date"][:4]

        movies.append(base + "-" + year)

        # ultra rare cases of multiple movies in the same year with the same title
        for i in range(expected_number_of_movies_from_same_year):
            movies.append(base + "-" + year + "-" + str(i+1))

        return movies

    services = []
    # ugly hack, might not always work!
    header = {"User-Agent": "Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"}
    possible = get_possible_urls(title, HEADERS)
    for p in possible:
        url = "https://www.justwatch.com/in/movie/" + p
        r = requests.get(url, headers=header)
        soup = BeautifulSoup(r.content, "html.parser")
        content = soup.find_all("div", class_="offer__content")

        for ser in content:
            pic = ser.find("picture")
            if pic:
                txt = pic.find("img")
                if txt["alt"] not in services:
                    services.append(txt["alt"])

    return {title: services}

def get_movie_reviews(title: str, n: int = 5) -> Dict[str, List[List[str]]]:
    """
    Get the reviews for a movie.

    Args:
        title: The title of the movie.
        n: Number of pages to be scraped.

    Returns:
        The reviews of all movies with that or similar title, as a list of lists.
    """
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
        
        year = results[0]["release_date"][:4]

        movies.append(base + "-" + year)

        # ultra rare cases of multiple movies in the same year with the same title
        for i in range(expected_number_of_movies_from_same_year):
            movies.append(base + "-" + year + "-" + str(i+1))

        return movies

    def strip_html_tags(text: str) -> str:
        return re.sub("<[^<]+?>", "", text)
    
    def sublist(list1: List[str], list2: List[List[str]]):
        if not list1 or not list2:
            return False
        for subl in list2:
            if not subl:
                continue
            el = subl[0]
        return el in list2
    
    def get_letterboxd_reviews(url: str, n: int) -> List[str]:
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

    possible = get_possible_urls(title, HEADERS)
    reviews = []
    for p in possible:
        url = ("https://letterboxd.com/film/" + p + "/reviews/by/activity")
        revs = get_letterboxd_reviews(url, n)
        if sublist(revs, reviews):
            continue
        if len(revs) > 0:
            reviews.append(revs)    

    return {title: reviews}

def get_person_credits(name: str) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Get the list of movies the person has worked on and additional information about each movie.

    Args:
        name: The name and surname of the person.

    Returns:
        The movies for each person with this name.
    """

    def get_credits_for_person(person_id, headers):
        url = "https://api.themoviedb.org/3/person/"+str(person_id)+"/movie_credits?language=en-US"
        response = requests.get(url, headers=headers)

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

    headers = {
            "accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjNzA3OTdmMTNkOGEyYjE2ODZhM2MxZTI0MzBmYmI1NCIsIm5iZiI6MTc0NDk4ODE2NC4yMjU5OTk4LCJzdWIiOiI2ODAyNjgwNDJjODVlNzk2NjM5OWJkYTYiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.prH_dcerfMwd_oxlbU6qBIaH5tqkBO3yu-z09XirBAE"
        }
    
    url = ("https://api.themoviedb.org/3/search/person?query="+ name + "&include_adult=false&language=en-US&page=1")
    response = requests.get(url, headers=headers)
    
    max_popularity = 0
    max_id = -1
    credits = {}
    for result in response.json()["results"]:
        if "id" in result.keys() and "popularity" in result.keys() and result["popularity"] > max_popularity:
            # if you want raw data add raw=True, there's a lot of junk in there though
            max_id = result["id"]
            max_popularity = int(result["popularity"])
    
    credits[name] = get_credits_for_person(max_id, headers)

    return credits

def get_movie_summary(title: str) -> Dict[str, Dict[str, Dict]]:
    """
    Get the plot summary of a movie.

    Args:
        title: The title of the movie.

    Returns:
        The plot summary of all movies with that or similar title, as a dict.
    """
    import warnings
    warnings.filterwarnings('ignore')
    orig_title = title
    headers = {
            "accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjNzA3OTdmMTNkOGEyYjE2ODZhM2MxZTI0MzBmYmI1NCIsIm5iZiI6MTc0NDk4ODE2NC4yMjU5OTk4LCJzdWIiOiI2ODAyNjgwNDJjODVlNzk2NjM5OWJkYTYiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.prH_dcerfMwd_oxlbU6qBIaH5tqkBO3yu-z09XirBAE"
        }

    movie = title.strip()
    #movie = movie.replace("\'", "%27")
    #movie = movie.replace("\'", "")
    out = {}
    
    url = ("https://api.themoviedb.org/3/search/movie?query=" + movie + "&include_adult=false&language=en-US&page=1")
    response = requests.get(url, headers=headers)
    results = response.json()["results"]
    if len(results) == 0:
        return {title: ""}
    
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
            out[orig_title] = {"summary": "", "plot": ""}
        else:
            out[orig_title] = {"summary": summary, "plot": plot} 

    return out

def get_similar_movies(title: str) -> Dict[str, List[str]]:
    """
    Get similar movies for a movie.

    Args:
        title: The title of the movie.

    Returns:
        The similar movies with that or similar title, as a list.
    """

    headers = {
            "accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjNzA3OTdmMTNkOGEyYjE2ODZhM2MxZTI0MzBmYmI1NCIsIm5iZiI6MTc0NDk4ODE2NC4yMjU5OTk4LCJzdWIiOiI2ODAyNjgwNDJjODVlNzk2NjM5OWJkYTYiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.prH_dcerfMwd_oxlbU6qBIaH5tqkBO3yu-z09XirBAE"
        }
    
    url = ("https://api.themoviedb.org/3/search/movie?query=" + title + "&include_adult=false&language=en-US&page=1")
    response = requests.get(url, headers=headers)
    results = response.json()["results"]
    if not results:
        return {title: []}
    id = results[0]["id"]

    url = "https://api.themoviedb.org/3/movie/"+str(id)+"/similar?language=en-US&page=1"
    response = requests.get(url, headers=headers)

    data = response.json()["results"]
    movies = []
    for d in data:
        movies.append(d["title"])

    return {title: movies}