import openai
import json
import pandas as pd
import tiktoken
import re

openai.api_key = "tu_api_key_aqui"

# Define tu universo de tags 
universo_tags = [
    "drama", "action", "friendship", "dark", "futurist", "romance", "comedy", "tragedy", "documental", "music", "thriller", "horror", "fantasy", "science fiction", "adventure", "mystery", "biography", "historical", "war", "western",
    "animation", "family", "crime", "noir", "superhero", "psychological", "sports", "music", "dance", "coming of age", "social issues", "political", "satire", "parody", "surrealism", "experimental", "cult classic", "independent", "blockbuster", "remake", "sequel", "franchise",
    "based on true story", "adaptation", "award-winning", "critically acclaimed"
    
]

# Cargar el CSV correcto
df = pd.read_csv('top_n_letterboxd.csv')

def contar_tokens(prompt, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))

def extraer_enrichment(reseñas, generos_actuales, temas_actuales):
    prompt = f"""
Lee las siguientes 15 reseñas de una película y responde SOLO en formato JSON con una lista de nuevos temas o géneros relevantes que no estén en las siguientes listas:
Géneros actuales: {', '.join(sorted(generos_actuales))}
Temas actuales: {', '.join(sorted(temas_actuales))}
Si no hay nuevos temas o géneros, responde una lista vacía [].
Reseñas:
{reseñas}
Ejemplo de respuesta:
["viajes en el tiempo", "existencialismo", "realismo mágico"]
"""
    tokens_usados = contar_tokens(prompt, model="gpt-4o-mini")
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.1
    )
    response_text = response.choices[0].message.content.strip()
    enrichment = extraer_lista_de_respuesta(response_text)
    if isinstance(enrichment, list):
        return enrichment, tokens_usados
    else:
        print("Respuesta inesperada del modelo (no es lista):")
        print(response_text.encode('ascii', errors='replace').decode())
        return [], tokens_usados

def extraer_json_de_respuesta(response_text):

    match = re.search(r'``[json\s*(\{.*?\})\s*](http://_vscodecontentref_/0)``', response_text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    if response_text.strip().startswith('{'):
        return json.loads(response_text.strip())
    match = re.search(r'(\{.*\})', response_text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return {}

def extraer_lista_de_respuesta(response_text):
    match = re.search(r'\[.*?\]', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    if response_text.strip().startswith('['):
        try:
            return json.loads(response_text.strip())
        except Exception:
            pass
    return {}

def puntuar_tags_pelicula(titulo, sinopsis, generos, temas, enrichment, universo_tags):
    prompt = f"""
Dado el siguiente universo de tags:
{json.dumps(universo_tags, ensure_ascii=False)}
Y la siguiente información de la película:
Título: {titulo}
Sinopsis: {sinopsis}
Géneros: {generos}
Temas: {temas}
Nuevos temas detectados: {', '.join(enrichment)}

Asigna un puntaje de 0 a 5 a cada tag según su relevancia para esta película, en caso de no tener relevancia simplemente puntúalo con un 0 y no lo agregues en el JSON. Responde solo en formato JSON:
{{
  "drama": 5,
  "action": 2,
  "friendship": 4
  // ...etc
}}
"""
    tokens_usados = contar_tokens(prompt, model="gpt-4o-mini")
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.1
    )
    response_text = response.choices[0].message.content.strip()
    tags_dict = extraer_json_de_respuesta(response_text)
    if tags_dict:
        return tags_dict, tokens_usados
    else:
        print("Respuesta inesperada del modelo:")
        print(response_text.encode('ascii', errors='replace').decode())
        return {}, tokens_usados

# Obtener géneros y temas actuales
generos_actuales = set()
temas_actuales = set()
for g in df['genres'].dropna():
    generos_actuales.update([x.strip().lower() for x in str(g).split(',')])
for t in df['themes'].dropna():
    temas_actuales.update([x.strip().lower() for x in str(t).split(',')])

enrichment_col = []
tokens_totales = 0
df_subset = df.head(10).copy()
for idx, fila in df_subset.iterrows():
    reseñas = str(fila.get('reviews', '')).split('|||')
    reseñas_texto = "\n".join([r.strip() for r in reseñas if r.strip()])
    print(f"Procesando enrichment para: {fila.get('title', '')}".encode('ascii', errors='replace').decode())
    enrichment, tokens_usados = extraer_enrichment(
        reseñas=reseñas_texto,
        generos_actuales=generos_actuales,
        temas_actuales=temas_actuales
    )
    print(f"Enrichment encontrado: {enrichment}".encode('ascii', errors='replace').decode())
    enrichment_col.append(enrichment)
    tokens_totales += tokens_usados

print(f"Tokens totales usados en enrichment: {tokens_totales}")

df_subset.loc[:, 'enrichment'] = enrichment_col

#flujo principal
tags_puntuados_col = []
resultados_json = []
for idx, fila in df_subset.iterrows():
    enrichment = enrichment_col[idx] if idx < len(enrichment_col) else []
    print(f"Procesando punteo de tags para: {fila.get('title', '')}".encode('ascii', errors='replace').decode())
    tags_puntuados, tokens_tags = puntuar_tags_pelicula(
        titulo=fila.get('title', ''),
        sinopsis=fila.get('synopsis', ''),
        generos=fila.get('genres', ''),
        temas=fila.get('themes', ''),
        enrichment=enrichment,
        universo_tags=universo_tags
    )
    tags_puntuados_col.append(tags_puntuados)
    tokens_totales += tokens_tags
    resultados_json.append({
        "title": fila.get('title', ''),
        "tags_puntuados": tags_puntuados,
        "tokens_usados": tokens_tags
    })

df_subset.loc[:, 'tags_puntuados'] = tags_puntuados_col

with open("tags_puntuados_peliculas.json", "w", encoding="utf-8") as f:
    json.dump(resultados_json, f, ensure_ascii=False, indent=2)

