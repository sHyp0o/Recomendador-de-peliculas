import requests
from bs4 import BeautifulSoup
import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import re

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/113.0.0.0 Safari/537.36"
    )
}

def fetch_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS)
    return BeautifulSoup(resp.text, "html.parser")

def extract_top_n_movies(n=5000) -> list:
    base_url = "https://letterboxd.com/films/by/rating/page/{}/"
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=chrome_options)

    movie_data = []
    page = 1
    while len(movie_data) < n:
        url = base_url.format(page)
        driver.get(url)
        time.sleep(4)
        posters = driver.find_elements(By.CSS_SELECTOR, "li.poster-container a.frame")
        if not posters:
            break  # No more movies/pages

        for poster in posters:
            href = poster.get_attribute("href")
            if href and "/film/" in href:
                movie_url = href
                movie_id = movie_url.split('/film/')[1].split('/')[0]
                try:
                    film_soup = fetch_soup(movie_url)
                    h1 = film_soup.find("h1", class_="headline-1")
                    title = h1.get_text(strip=True) if h1 else None
                    year_tag = film_soup.select_one('a[href^="/films/year/"]')
                    year = year_tag.get_text(strip=True) if year_tag else None
                    movie_data.append({
                        "title": title,
                        "year": year,
                        "url": movie_url,
                        "id": movie_id
                    })
                    print(f"Extraído: {title} ({year}) con ID {movie_id}")
                    time.sleep(1.2)
                    if len(movie_data) >= n:
                        break
                except Exception as e:
                    print(f"Error al procesar {movie_url}: {e}")
                    continue
        page += 1

    driver.quit()
    return movie_data



def extract_rating_from_film_page(url: str) -> str:
    soup = fetch_soup(url)
    script_tag = soup.find("script", type="application/ld+json")
    if not script_tag or not script_tag.string:
        return None
    try:
        import json
        json_data = json.loads(script_tag.string.strip())
        rating = json_data.get("aggregateRating", {}).get("ratingValue")
        return str(rating) if rating else None
    except Exception:
        return None

def extract_rating_and_synopsis_from_film_page(url: str):
    soup = fetch_soup(url)
    score = None
    synopsis = None

    #RATING
    ld_json = soup.find("script", type="application/ld+json")
    if ld_json and ld_json.string:
        try:
            import json
            json_str = ld_json.string
            json_str = json_str.replace("/* <![CDATA[ */", "").replace("/* ]]> */", "").strip()
            data = json.loads(json_str)
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and "aggregateRating" in entry:
                        agg_rating = entry["aggregateRating"]
                        score_val = agg_rating.get("ratingValue")
                        if score_val is not None:
                            score = str(score_val)
                            break
            elif isinstance(data, dict):
                agg_rating = data.get("aggregateRating", {})
                score_val = agg_rating.get("ratingValue")
                score = str(score_val) if score_val is not None else None
        except Exception:
            score = None

    #SINOPSIS
    desc = soup.find("meta", {"name": "description"})
    synopsis = desc["content"].split("—")[0].strip() if desc and "content" in desc.attrs else None

    #GÉNEROS
    genre_tags = soup.select('a[href^="/films/genre/"]')
    genres = [g.get_text(strip=True) for g in genre_tags]

    #TEMAS
    theme_tags_1 = soup.select('a[href^="/films/theme/"]')
    theme_tags_2 = soup.select('a[href^="/films/mini-theme/"]')
    themes = [t.get_text(strip=True) for t in theme_tags_1 + theme_tags_2 if t.get_text(strip=True)]

    return score, synopsis, genres, themes

def extract_reviews_from_film_page(url: str, max_reviews=15):
    if not url.endswith('/'):
        url += '/'
    reviews_url = url + "reviews/by/added/"

    try:
        soup = fetch_soup(reviews_url)
    except Exception as e:
        print(f"Error al obtener reseñas de {reviews_url}: {e}")
        return []

    reviews = []
    review_blocks = soup.select("div.listitem")

    for block in review_blocks:
        # Texto de la reseña
        text_tag = block.select_one(".body-text, .review-text, p")
        text = text_tag.get_text(strip=True) if text_tag else None

        # Nombre del usuario
        user_tag = block.select_one("a.avatar img")
        username = user_tag["alt"] if user_tag and user_tag.has_attr("alt") else None

        # Puntuación del usuario
        rating_tag = block.select_one(".rating")
        user_rating = None
        if rating_tag:
            classes = rating_tag.get("class", [])
            for c in classes:
                if c.startswith("rated-"):
                    try:
                        user_rating = int(c.replace("rated-", "")) / 10  
                    except Exception:
                        user_rating = None

        if text:
            reviews.append({
                "username": username,
                "text": text,
                "user_rating": user_rating
            })

        if len(reviews) >= max_reviews:
            break

    return reviews



def main():
    movies = extract_top_n_movies()
    print(f"Se encontraron {len(movies)} películas. Obteniendo ratings...")

    for i, movie in enumerate(movies):
        print(f"[{i+1}/{len(movies)}] {movie['title']} ({movie['year']})".encode('ascii', errors='replace').decode())
        rating, synopsis, genres, themes = extract_rating_and_synopsis_from_film_page(movie["url"])
        movie["rating"] = rating
        movie["synopsis"] = synopsis
        movie["genres"] = ", ".join(genres)
        movie["themes"] = ", ".join(themes)
        reviews = extract_reviews_from_film_page(movie["url"], max_reviews=15)
        movie["reviews"] = " ||| ".join([
            f'{r["username"]} ({r["user_rating"] if r["user_rating"] is not None else "N/A"}): {r["text"]}'
            for r in reviews
        ])
        time.sleep(1.5)

    with open("top_n_letterboxd.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id","title", "year", "url", "rating", "synopsis", "genres", "themes", "reviews"])  # Añadir 'id'
        writer.writeheader()
        writer.writerows(movies)


    print("Archivo CSV guardado como top_n_letterboxd.csv")

if __name__ == "__main__":
    main()
