import requests
from bs4 import BeautifulSoup
import time
import csv
import unicodedata
import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0"}

def limpiar_texto(texto):
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    texto = texto.replace("ñ", "n").replace("Ñ", "N")
    return texto

# Cargar IDs de películas (slugs) del CSV
df = pd.read_csv("top_n_letterboxd.csv")
titulos_peliculas = set(df['id'].dropna().str.strip()) 

def get_popular_usernames(n=1000):
    usernames = set()
    page = 1
    while len(usernames) < n:
        url = f"https://letterboxd.com/members/popular/this/week/page/{page}/"
        print(f"Accediendo a: {url}")
        res = requests.get(url, headers=HEADERS)
        if res.status_code != 200:
            print(f"Error {res.status_code} en la pagina {page}")
            break
        soup = BeautifulSoup(res.text, "html.parser")
        for a in soup.select("a.avatar"):
            href = a.get("href")
            if href:
                username = href.strip("/").split("/")[0]
                usernames.add(username)
                if len(usernames) >= n:
                    break
        page += 1
        time.sleep(1)
    return list(usernames)

def get_reviews_from_activity(username, titulos_peliculas):
    reviews = []
    page = 1
    while True:
        url = f"https://letterboxd.com/{username}/films/reviews/page/{page}/"
        resp = requests.get(url)
        if resp.status_code != 200:
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        review_blocks = soup.select("div.listitem.js-listitem")
        if not review_blocks:
            break
        for block in review_blocks:
            poster_div = block.select_one("div[data-film-slug]")
            film_id = poster_div["data-film-slug"] if poster_div and poster_div.has_attr("data-film-slug") else ""
            film_tag = block.select_one("h2.name a")
            titulo = film_tag.text.strip() if film_tag else ""

            # Comparar usando el film_id (slug) extraído y los IDs del CSV
            if film_id in titulos_peliculas:
                rating_tag = block.select_one("span.rating")
                puntuacion = None
                if rating_tag:
                    for c in rating_tag.get("class", []):
                        if c.startswith("rated-"):
                            valor = c.replace("rated-", "")
                            if len(valor) == 1:
                                puntuacion = float(valor)
                            else:
                                puntuacion = float(valor) / 2
                # Extraer reseña
                review_tag = block.select_one("div.js-review-body")
                reseña = review_tag.get_text(strip=True) if review_tag else ""
                reviews.append({
                    "usuario": username,
                    "pelicula": titulo,
                    "film_id": film_id,
                    "puntuacion": puntuacion,
                    "resena": reseña
                })
        page += 1
    return reviews

def guardar_csv(datos, archivo="resenas_letterboxd.csv"):
    with open(archivo, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["film_id","usuario", "pelicula", "puntuacion", "resena"])
        writer.writeheader()
        writer.writerows(datos)
    print(f"\nCSV guardado en: {archivo} con {len(datos)} filas.")

def main():
    print("Obteniendo usuarios populares...")
    usuarios = get_popular_usernames(1000) #puedes cambiar este parametro para aumentar o disminuir el numero de usuarios unicos a scrapear 
    print(f"Total de usuarios obtenidos: {len(usuarios)}")

    total = 0
    datos = []

    for idx, usuario in enumerate(usuarios, 1):
        print(f"\n[{idx}] Extrayendo resenas de: {usuario}")
        
        resenas = get_reviews_from_activity(usuario, titulos_peliculas)
        if len(resenas) >= 1:
            print(f"{usuario} tiene {len(resenas)} resenas publicas.")
            for i, r in enumerate(resenas, 1):
                print(f"\nResena {i}:")
                print("Titulo:", limpiar_texto(r['pelicula']))
                print("Puntuacion:", r['puntuacion'])
                print("Texto:", limpiar_texto(r['resena'][:200]), "...")
            datos.extend(resenas)
            total += len(resenas)
        else:
            print(f"{usuario} no tiene suficientes resenas publicas.")
        time.sleep(1)

    guardar_csv(datos)
    print(f"\nFinalizado. Total de resenas extraidas: {total}")

if __name__ == "__main__":
    main()
