import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm


def get_movie_data(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, features="html.parser")

    movie_description = soup.find_all(
        "div", class_="movie_synopsis clamp clamp-6 js-clamp"
    )[0].text.strip()
    movie_description_clean = re.sub(r"[\n| ]+", " ", movie_description)

    movie_genre = soup.find_all("div", class_="meta-value genre")[0].text.strip()
    movie_genre_clean = re.sub(r"[\n| ]+", " ", movie_genre)

    movie_name = soup.find_all("h1", class_="scoreboard__title")[0].text.strip()
    movie_name_clean = re.sub(r"[\n| ]+", " ", movie_name)

    return movie_name_clean, movie_genre_clean, movie_description_clean


def get_movie_urls(html_page):
    with open(html_page, "r") as html:
        res = html.read()

    soup = BeautifulSoup(res, features="html.parser")

    movies_grid = soup.find_all("div", class_="discovery-tiles__wrap")[1]
    movie_urls = movies_grid.find_all("a", class_="js-tile-link", href=True)

    return [film["href"] for film in movie_urls]


if __name__ == "__main__":
    movie_urls = get_movie_urls("movie_300.html")
    output_file = "data_v2.csv"

    with open(output_file, "a") as data:
        data.write("|".join(["movie_name", "movie_genre", "movie_description"]) + "\n")

    for movie_url in tqdm(movie_urls):
        try:
            movie_data = get_movie_data(f"https://www.rottentomatoes.com{movie_url}")

            with open(output_file, "a") as data:
                data.write("|".join(movie_data) + "\n")
        except:
            continue
