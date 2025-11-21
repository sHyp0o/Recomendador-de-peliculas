import json
import csv
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

with open(SCRIPT_DIR / "tags_puntuados_sr.json", "r", encoding="utf-8") as f:
    data = json.load(f)

titulos = [d.get("title", f"Pelicula {i+1}") for i, d in enumerate(data)]
tags_dicts = [d.get("tags_puntuados", {}) for d in data]

universo_tags = sorted({tag for d in tags_dicts for tag in d.keys()})
X = np.array([[tags.get(tag, 0) for tag in universo_tags] for tags in tags_dicts])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

def closest_centroid(point, centroids):
    distances = np.sum((centroids - point) ** 2, axis=1)
    return np.argmin(distances)

def calculate_prediction_strength(X_test, test_labels, train_centroids, k):
    cluster_indices = np.where(test_labels == k)[0]
    n_k = len(cluster_indices)
    if n_k <= 1:
        return 1.0
    same_cluster_count = 0
    total_pairs = n_k * (n_k - 1) // 2
    for i in range(len(cluster_indices)):
        for j in range(i + 1, len(cluster_indices)):
            p1 = X_test[cluster_indices[i]]
            p2 = X_test[cluster_indices[j]]
            if closest_centroid(p1, train_centroids) == closest_centroid(p2, train_centroids):
                same_cluster_count += 1
    return same_cluster_count / total_pairs

def prediction_strength_for_k(X_test, test_labels, train_centroids, k):
    strengths = []
    for cluster_id in range(k):
        ps = calculate_prediction_strength(X_test, test_labels, train_centroids, cluster_id)
        strengths.append(ps)
    return min(strengths) if strengths else 0.0

def compute_prediction_strength(X, max_k=15, n_trials=10, test_size=0.2, random_state=42):
    max_k = min(max_k, len(X) - 1)
    ks = range(2, max_k + 1)
    all_strengths = {k: [] for k in ks}
    for trial in range(n_trials):
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state + trial)
        for k in ks:
            if k > min(len(X_train), len(X_test)):
                all_strengths[k].append(0.0)
                continue
            kmeans_train = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans_test = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans_train.fit(X_train)
            kmeans_test.fit(X_test)
            ps = prediction_strength_for_k(X_test, kmeans_test.labels_, kmeans_train.cluster_centers_, k)
            all_strengths[k].append(ps)
    mean_strengths = [np.mean(all_strengths[k]) for k in ks]
    std_strengths = [np.std(all_strengths[k]) for k in ks]
    return list(ks), mean_strengths, std_strengths

def select_optimal_k(ks, mean_strengths, threshold=0.8):
    optimal_k = 2
    for k, ps in zip(ks, mean_strengths):
        if ps >= threshold:
            optimal_k = k
    return optimal_k

def get_central_movie(X, clusters, centroids, titulos):
    central_movies = {}
    for c in sorted(set(clusters)):
        cluster_indices = np.where(clusters == c)[0]
        centroid = centroids[c]
        distances = np.sqrt(np.sum((X[cluster_indices] - centroid) ** 2, axis=1))
        closest_local_idx = np.argmin(distances)
        closest_global_idx = cluster_indices[closest_local_idx]
        central_movies[c] = {
            'indice': closest_global_idx,
            'titulo': titulos[closest_global_idx],
            'distancia': distances[closest_local_idx]
        }
    return central_movies

print("Calculando Prediction Strength...")
ks, mean_ps, std_ps = compute_prediction_strength(X, max_k=15, n_trials=10)

optimal_k = select_optimal_k(ks, mean_ps, threshold=0.8)
print(f"\nK óptimo seleccionado (umbral 0.8): {optimal_k}")
print(f"Prediction Strength para k={optimal_k}: {mean_ps[ks.index(optimal_k)]:.3f}")

print("\nResumen de Prediction Strength por k:")
for k, ps, std in zip(ks, mean_ps, std_ps):
    marker = " <-- óptimo" if k == optimal_k else ""
    print(f"  k={k}: PS={ps:.3f} ± {std:.3f}{marker}")

n_clusters = optimal_k
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

print(f"\n--- Métricas de Evaluación (k={n_clusters}) ---")
sil_score = silhouette_score(X, clusters)
calinski_score = calinski_harabasz_score(X, clusters)
davies_score = davies_bouldin_score(X, clusters)

print(f"Silhouette Score: {sil_score:.3f}")
print(f"Calinski-Harabasz Index: {calinski_score:.3f}")
print(f"Davies-Bouldin Index: {davies_score:.3f}")

cluster_names = {}
for c in sorted(set(clusters)):
    subset_idx = [i for i, cl in enumerate(clusters) if cl == c]
    all_tags = []
    for idx in subset_idx:
        all_tags.extend([tag for tag, score in tags_dicts[idx].items() if score > 0])
    comunes = [feat for feat, _ in Counter(all_tags).most_common(2)]
    cluster_names[c] = ", ".join(comunes) if comunes else "Sin tags destacados"

central_movies = get_central_movie(X, clusters, kmeans.cluster_centers_, titulos)

# ============================================================
# GUARDAR RESULTADOS EN CSV
# ============================================================
csv_filename = "datos_agrupamiento_sin_enrichment.csv"

with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Sección 1: K óptimo y Prediction Strength
    writer.writerow(['K óptimo', optimal_k])
    writer.writerow(['Prediction Strength k óptimo', round(mean_ps[ks.index(optimal_k)], 3)])
    writer.writerow([])
    
    # Sección 2: Prediction Strength por k
    writer.writerow(['k', 'PS', 'std'])
    for k, ps, std in zip(ks, mean_ps, std_ps):
        writer.writerow([k, round(ps, 3), round(std, 3)])
    writer.writerow([])
    
    # Sección 3: Métricas de evaluación
    writer.writerow(['Métrica', 'Valor'])
    writer.writerow(['Silhouette Score', round(sil_score, 3)])
    writer.writerow(['Calinski-Harabasz Index', round(calinski_score, 3)])
    writer.writerow(['Davies-Bouldin Index', round(davies_score, 3)])
    writer.writerow([])
    
    # Sección 4: Resumen de clusters
    writer.writerow(['Cluster', 'Nombre', 'Num películas', 'Película central', 'Distancia centroide'])
    for c in sorted(set(clusters)):
        n_elementos = sum(1 for cl in clusters if cl == c)
        central = central_movies[c]
        writer.writerow([
            c,
            cluster_names[c],
            n_elementos,
            central['titulo'],
            round(central['distancia'], 4)
        ])

print(f"\n✓ Datos guardados en '{csv_filename}'")
print(f"  - Total de películas: {len(titulos)}")
print(f"  - Columnas: indice, titulo, cluster, cluster_nombre, es_central, distancia_centroide, pca_1, pca_2, tags")

# Resumen Final
print(f"\n--- Resumen de Clusters (k={n_clusters}) ---")
for c in sorted(set(clusters)):
    n_elementos = sum(1 for cl in clusters if cl == c)
    central = central_movies[c]
    print(f"\nCluster {c} ({cluster_names[c]}) - {n_elementos} películas:")
    print(f"  Película central: '{central['titulo']}' (índice: {central['indice']})")
    print(f"  Distancia al centroide: {central['distancia']:.4f}")
    print(f"  Tags: {tags_dicts[central['indice']]}")
    print("-" * 60)

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].errorbar(ks, mean_ps, yerr=std_ps, marker='o', capsize=5, capthick=2)
axes[0].axhline(y=0.8, color='r', linestyle='--', label='Umbral 0.8')
axes[0].axhline(y=0.9, color='g', linestyle=':', label='Umbral 0.9')
axes[0].set_xlabel('Número de clusters (k)')
axes[0].set_ylabel('Prediction Strength')
axes[0].set_title('Prediction Strength: Tags sin enrichment')
axes[0].set_xticks(ks)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

centroids_2d = pca.transform(kmeans.cluster_centers_)
scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', s=50, alpha=0.7)
axes[1].set_xlabel('PCA 1')
axes[1].set_ylabel('PCA 2')
axes[1].set_title(f'Clustering (k={n_clusters}, PS={mean_ps[ks.index(optimal_k)]:.3f})')
axes[1].legend(*scatter.legend_elements(), title="Clustering: Tags sin enrichment")

for i, (x, y) in enumerate(centroids_2d):
    axes[1].annotate(cluster_names[i], (x, y), fontsize=8, weight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

for c, info in central_movies.items():
    idx = info['indice']
    x, y = X_pca[idx]
    axes[1].scatter(x, y, c='red', s=200, marker='*', edgecolors='black', linewidths=1, zorder=5)
    titulo_corto = info['titulo'][:15] + '...' if len(info['titulo']) > 15 else info['titulo']
    axes[1].annotate(titulo_corto, (x, y), fontsize=7,
                     xytext=(5, 5), textcoords='offset points',
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

plt.tight_layout()
plt.show()