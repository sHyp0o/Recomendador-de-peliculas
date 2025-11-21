# Recomendador-de-peliculas
Recomendador de peliculas hibridas haciendo uso de llms y representación semántica mediante etiquetas

Autores
Vicente Eugenio Salinas Guerra,
Gabriel Andrés Pizarro Rivera 

Requisitos
- Python 3.12

Orden de ejecución
1. Ejecutar primero el scraping de películas (webscrap_peliculas.py)
2. Luego ejecutar el scraping de usuarios (webscrap_users.py)
3. Enrichment (enrichment.py)
4. Agrupamiento sin enrichment (agrupamiento_sin_enrichment.py)
5. Agrupamiento con enrichment (agrupamiento_con_enrichment.py)
6. Recomendacion híbrida (filtrado_híbrido.py)

Motivo
Al extraer usuarios se seleccionan N usuarios únicos que hayan visto al menos una película incluida en top_n_letterboxd.csv, 
por lo que el scraping de películas debe correrse antes.

Modos de Recomendacion (filtrado_híbrido.py)

Para realizar recomendaciones con o sin enriquecimiento solo debe reemplazar la variable a continuación según su necesidad:

- Con enriquecimiento: TAGS_JSON = "tags_puntuados_peliculas.json"
- Sin enriquecimiento: TAGS_JSON = "tags_puntuados_sr.json"

Significado de los archivos
- tags_puntuados_peliculas.json: tags puntuados con enriquecimiento
- tags_puntuados_sr.json: tags puntuados sin enriquecimiento
- top_n_letterboxd.csv: lista de películas objetivo usada para filtrar usuarios
- datos_agrupamiento_enrichment.csv: muestra valores relacionados a la reducción de dimensionalidad y el número de clústers para los tags con enriquecimiento.
- datos_agrupamiento_sin_enrichment.csv: muestra valores relacionados a la reducción de dimensionalidad y el número de clústers para los tags sin enriquecimiento.

Notas
- Asegúrate de usar Python 3.12.
- Mantén el orden de ejecución indicado para obtener resultados correctos
- Con los datasets adjuntos puedes ejecutar directamente el filtrado híbrido; Asi los otros tres pasos anteriores al filtrado son opcionales y se incluyen solo para fines evaluativos o para ajustar parámetros si es necesario.
