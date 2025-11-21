
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from pathlib import Path


CSV_PATH = "resenas_letterboxd.csv"
TAGS_JSON = "tags_puntuados_peliculas.json"


USERS_TO_RUN = [
    "deathproof","timtamtitus","alexil","andredenervaux","bishoyy","caityperrie","chatterb0xd",
    "drewgregory","dumbsville","elisavetkei","filmcal","funeralroses","hamburgerpimp","happypumpkin18",
    "hugefilmguy","hzjoett","iancurtys","ianjonas","indiasfilms","jeaba","jizzmonkey","kmokler",
    "lautengcokj","lilusions","m__ollymac","meteorrejector","nerdzola","oppenmommy","paulscheer",
    "siena_brown","simonramshaw","staceyerinm","tedsmovies","thegaychingy","thejoelynch","thejoshl",
    "timtamtitus","zascmo"
]
N_RECS = 30
ALPHA = 0.5  # peso para combinacion colaborativo/contenido
UMBRAL_PERFIL = 1  # >= para construir perfil de tags (puntuacion minima)

def normalizar(t):
    try:
        from unidecode import unidecode
        return unidecode(str(t).strip().lower())
    except Exception:
        return str(t).strip().lower()

df = pd.read_csv(CSV_PATH)
tabla_raw = df.pivot_table(index='usuario', columns='pelicula', values='puntuacion')
tabla_filled = tabla_raw.fillna(0)

# Cargar tags JSON y crear diccionario normalizado
tags_data = []
if Path(TAGS_JSON).exists():
    with open(TAGS_JSON, "r", encoding="utf-8") as f:
        tags_data = json.load(f)
tags_por_pelicula = {}
for d in tags_data:
    title = d.get("title")
    if not title:
        continue
    tags_por_pelicula[normalizar(title)] = d.get("tags_puntuados", {})

universo_tags = sorted({tag for d in tags_data if "tags_puntuados" in d for tag in d["tags_puntuados"].keys()})
if not universo_tags:
    # fallback: lista pequeña conocida si JSON no contiene tags
    universo_tags = [
        "drama", "action", "friendship", "dark", "futurist", "romance", "comedy", "tragedy", "documental", "music", "thriller", "horror", "fantasy", "science fiction", "adventure", "mystery", "biography", "historical", "war", "western",
        "animation", "family", "crime", "noir", "superhero", "psychological", "sports", "music", "dance", "coming of age", "social issues", "political", "satire", "parody", "surrealism", "experimental", "cult classic", "independent", "blockbuster", "remake", "sequel", "franchise",
        "based on true story", "adaptation", "award-winning", "critically acclaimed"
    ]

results = {}

for usuario in USERS_TO_RUN:
    if usuario not in tabla_filled.index:
        print(f"Usuario no encontrado, se omite: {usuario}")
        continue

    #HOLDOUT SIMPLE: separar un pequeño conjunto de "relevantes" (test) desde las vistas del usuario
    vistas_todas = tabla_raw.columns[tabla_raw.loc[usuario].notna()]
    if len(vistas_todas) == 0:
        print(f"  Usuario {usuario} no tiene vistas -> se omite holdout")
        test_items = []
    else:
        np.random.seed(42)
        FRACTION = 0.3
        MIN_TEST = 5
        n = len(vistas_todas)
        if n <= 1:
            test_items = []  # no se puede hacer holdout sensato
        else:
            desired = max(MIN_TEST, int(n * FRACTION))
            # no tomar más que la población; dejar al menos 1 item para entrenamiento si es posible
            test_size = min(n - 1, desired)
            test_items = list(np.random.choice(vistas_todas, size=test_size, replace=False))

    # trabajar con copias donde los test_items se marcan como "no vistos" para la generación de recomendaciones
    tabla_filled_tmp = tabla_filled.copy()
    tabla_raw_tmp = tabla_raw.copy()
    for t in test_items:
        tabla_filled_tmp.loc[usuario, t] = 0
        tabla_raw_tmp.loc[usuario, t] = np.nan

    # vector usuario para colaborativo
    usuario_vec = tabla_filled_tmp.loc[usuario].values.reshape(1, -1)
    similitudes = cosine_similarity(usuario_vec, tabla_filled_tmp.values)[0]

    # pelis no vistas
    pelis_no_vistas = tabla_raw_tmp.columns[tabla_raw_tmp.loc[usuario].isna()]

    # predicción colaborativa
    sim_scores = {}
    denom = (np.abs(similitudes).sum() + 1e-8)
    for peli in pelis_no_vistas:
        puntuaciones = tabla_filled[peli]
        pred = float(np.dot(similitudes, puntuaciones) / denom)
        sim_scores[peli] = pred

    # construir perfil de tags del usuario (películas puntuadas >= UMBRAL_PERFIL)
    pelis_perfil = tabla_raw.columns[tabla_raw.loc[usuario] >= UMBRAL_PERFIL]
    usuario_tags_vec = np.zeros(len(universo_tags))
    for peli in pelis_perfil:
        tags = tags_por_pelicula.get(normalizar(peli), {})
        tag_vec = np.array([tags.get(tag, 0) for tag in universo_tags])
        usuario_tags_vec += tag_vec
    usuario_tags_vec = usuario_tags_vec.reshape(1, -1)

    # similitud de contenido (tags) para pelis no vistas
    sim_tags = {}
    for peli in pelis_no_vistas:
        tags = tags_por_pelicula.get(normalizar(peli), {})
        peli_vec = np.array([tags.get(tag, 0) for tag in universo_tags]).reshape(1, -1)
        sim = float(cosine_similarity(usuario_tags_vec, peli_vec)[0][0]) if usuario_tags_vec.sum() != 0 else 0.0
        sim_tags[peli] = sim



   
    top_colab = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    top_cont  = sorted(sim_tags.items(), key=lambda x: x[1], reverse=True)[:20]

    vals_cont = list(sim_tags.values())
    
    # si sim_tags es todo ceros, combinadas serán escala de colaborativo y orden no cambia
    if all(v == 0 for v in vals_cont):
        print("  AVISO: sim_tags son todos 0 → recomendaciones combinadas mantendrán el orden colaborativo (siendo proporcionales).")

    # top-N por metodo
    recomendadas_colab = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)[:N_RECS]
    recomendadas_cont = sorted(sim_tags.items(), key=lambda x: x[1], reverse=True)[:N_RECS]

    # combinado
    final_scores = {}
    for peli in pelis_no_vistas:
        final_scores[peli] = ALPHA * sim_scores.get(peli, 0) + (1 - ALPHA) * sim_tags.get(peli, 0)
    recomendadas_comb = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:N_RECS]

    # relevantes: calcular similitud con el perfil sobre TODO el catálogo (vistas y no vistas)
    final_scores_all = {}
    for peli in tabla_raw.columns:
        puntuaciones_all = tabla_filled[peli]
        pred_all = float(np.dot(similitudes, puntuaciones_all) / denom)

        # similitud de contenido (tags) sobre todo el catálogo
        tags = tags_por_pelicula.get(normalizar(peli), {})
        peli_vec = np.array([tags.get(tag, 0) for tag in universo_tags]).reshape(1, -1)
        sim_tag_all = float(cosine_similarity(usuario_tags_vec, peli_vec)[0][0]) if usuario_tags_vec.sum() != 0 else 0.0

        final_scores_all[peli] = ALPHA * pred_all + (1 - ALPHA) * sim_tag_all

    # seleccionar relevantes: umbral por percentil, fallback a top-k si no hay suficientes
    scores_arr = np.array(list(final_scores_all.values()))
    if scores_arr.size == 0:
        relevantes_list = []
    else:
        mu = float(scores_arr.mean())
        sigma = float(scores_arr.std())
        beta = 0.5  # ajustar: mayor beta => menos relevantes
        candidatos = []
        if sigma > 1e-8:
            thresh = mu + beta * sigma
            candidatos = [p for p, s in final_scores_all.items() if s >= thresh]

        if not candidatos:
            min_relevantes = 5
            max_relevantes = 200
            perfil_size = max(1, int(tabla_raw.loc[usuario].notna().sum()))
            # top base = 5% del catálogo + una parte proporcional al tamaño del perfil
            base_top = max(1, int(0.05 * len(final_scores_all)))
            top_k = min(max_relevantes, max(min_relevantes, base_top + perfil_size // 2))
            top_k = min(top_k, len(final_scores_all))
            candidatos = [p for p, _ in sorted(final_scores_all.items(), key=lambda x: x[1], reverse=True)[:top_k]]

        relevantes_list = candidatos

    results[usuario] = {
        "colaborativo": [p for p, _ in recomendadas_colab],
        "contenido": [p for p, _ in recomendadas_cont],
        "combinado": [p for p, _ in recomendadas_comb],
        "test": [p for p in test_items],
        "relevantes": [p for p in relevantes_list],  # relevantes según similitud al perfil (vistas o no)
         # vistas de entrenamiento: vistas originales menos los items del holdout
         "vistas_train": [p for p in tabla_raw_tmp.columns[tabla_raw_tmp.loc[usuario].notna()]]
    }

    # imprimir resumen conciso
    print(f"\nUsuario: {usuario}")
    print("  Colab:", results[usuario]["colaborativo"])
    print("  Cont :", results[usuario]["contenido"])
    print("  Comb :", results[usuario]["combinado"])


OUT = "recomendaciones_por_usuario.json"
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nRecomendaciones guardadas en: {OUT}")

# ------------------ evaluación (por usuario) ------------------
def precision_at_k(recs, relevantes, k):
    if k == 0: return 0.0
    return len(set(recs[:k]) & set(relevantes)) / k

def recall_at_k(recs, relevantes, k):
    if not relevantes: return 0.0
    return len(set(recs[:k]) & set(relevantes)) / len(relevantes)

import math
def ndcg_at_k(recs, relevantes, k):
    dcg = 0.0
    for i, r in enumerate(recs[:k]):
        if r in relevantes:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevantes), k)))
    return (dcg / idcg) if idcg > 0 else 0.0

def cobertura(recs_dict, total_peliculas):
    peliculas_recomendadas = set()
    for recs in recs_dict.values():
        peliculas_recomendadas.update(recs)
    return len(peliculas_recomendadas) / max(1, total_peliculas)

def serendipia(recs, relevantes, populares, vistas_usuario):
    if not recs: return 0.0
    inesperadas = [r for r in recs if r not in populares and r not in vistas_usuario]
    return len(set(inesperadas) & set(relevantes)) / len(recs)

# Parámetros de evaluación
K = N_RECS  # usar mismo N_RECS para metrics@K
TOTAL_PELICULAS = len(tabla_raw.columns)

populares_global = set(normalizar(p) for p in tabla_filled.sum().sort_values(ascending=False).head(100).index)

per_user_metrics = {}

for usuario in results:
    if usuario not in tabla_raw.index:
        continue

    # Preferir las vistas de entrenamiento (sin los items del holdout) guardadas en results
    if results[usuario].get("vistas_train"):
        vistas_usuario = set(normalizar(p) for p in results[usuario]["vistas_train"])
    else:
        vistas_usuario = set(normalizar(p) for p in tabla_raw.columns[tabla_raw.loc[usuario].notna()])

    relevantes_calc = set(normalizar(p) for p in results[usuario].get("relevantes", []))
    relevantes_no_vistas = relevantes_calc  

    recs_colab = [normalizar(p) for p in results[usuario]["colaborativo"]]
    recs_cont  = [normalizar(p) for p in results[usuario]["contenido"]]
    recs_comb  = [normalizar(p) for p in results[usuario]["combinado"]]

    metrics_user = {}

    for name, recs in [("colab", recs_colab), ("cont", recs_cont), ("comb", recs_comb)]:
        prec = precision_at_k(recs, relevantes_no_vistas, K)
        rec = recall_at_k(recs, relevantes_no_vistas, K)
        ndcg = ndcg_at_k(recs, relevantes_no_vistas, K)
        ser = serendipia(recs, relevantes_no_vistas, populares_global, vistas_usuario)
        metrics_user[name] = {
            "precision@k": float(prec),
            "recall@k": float(rec),
            "ndcg@k": float(ndcg),
            "serendipia": float(ser),
            "k": int(K),
            "num_relevantes": int(len(relevantes_no_vistas))
        }

    per_user_metrics[usuario] = metrics_user

    # imprimir métricas del usuario
    print(f"\nUsuario: {usuario}")
    for method in ("colab", "cont", "comb"):
        m = metrics_user[method]
        print(f"  [{method.upper()}] Precision@{m['k']}: {m['precision@k']:.4f} | Recall@{m['k']}: {m['recall@k']:.4f} | NDCG@{m['k']}: {m['ndcg@k']:.4f} | Serendipia: {m['serendipia']:.4f} | #relevantes: {m['num_relevantes']}")

# Adjuntar métricas por usuario al archivo de salida de recomendaciones
OUT_METRICS = "recomendaciones_por_usuario_con_metricas.json"
combined_output = {}
for usuario in results:
    combined_output[usuario] = {
        "recomendaciones": results.get(usuario, {}),
        "metricas": per_user_metrics.get(usuario, {})
    }

with open(OUT_METRICS, "w", encoding="utf-8") as f:
    json.dump(combined_output, f, ensure_ascii=False, indent=2)

print(f"\nMétricas por usuario guardadas en: {OUT_METRICS}")

# calcular promedios globales de varias métricas
agg = {name: {"precision": [], "recall": [], "ndcg": [], "serendipia": []} for name in ("colab","cont","comb")}
for usuario, mets in per_user_metrics.items():
    for name in ("colab","cont","comb"):
        m = mets.get(name, {})
        agg[name]["precision"].append(m.get("precision@k", 0.0))
        agg[name]["recall"].append(m.get("recall@k", 0.0))
        agg[name]["ndcg"].append(m.get("ndcg@k", 0.0))
        agg[name]["serendipia"].append(m.get("serendipia", 0.0))

print("\nResumen de métricas promedio (por método):")
for name in ("colab","cont","comb"):
    p_mean = float(np.mean(agg[name]["precision"])) if agg[name]["precision"] else 0.0
    r_mean = float(np.mean(agg[name]["recall"])) if agg[name]["recall"] else 0.0
    n_mean = float(np.mean(agg[name]["ndcg"])) if agg[name]["ndcg"] else 0.0
    s_mean = float(np.mean(agg[name]["serendipia"])) if agg[name]["serendipia"] else 0.0
    print(f"  {name.upper()}: Precision@{K}: {p_mean:.4f} | Recall@{K}: {r_mean:.4f} | NDCG@{K}: {n_mean:.4f} | Serendipia: {s_mean:.4f}")