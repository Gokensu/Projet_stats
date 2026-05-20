import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
import pandas as pd

# ================================
# 1.2 Collecte des données
# ================================

# Lecture du fichier CSV contenant les réponses du sondage
df = pd.read_csv("csv/sondage.csv")

# Renommage des colonnes pour avoir des noms plus simples et plus clairs
df.columns = ["horodateur", "td", "heures", "raisons"]

# Dictionnaire associant chaque intervalle d'heures de sommeil
# à une valeur numérique représentant son point milieu
MILIEUX = {
    "Moins de 4h"    : 3.0,
    "Entre 4h et 6h" : 5.0,
    "Entre 6h et 8h" : 7.0,
    "Plus de 8h"     : 9.0,
}

# Conversion des réponses textuelles en valeurs numériques
# grâce au dictionnaire MILIEUX
DATA = df["heures"].map(MILIEUX).values

# Exemple du tableau obtenu :
# [7, 5, 5, 5, 7, 7, 7, 5, 5, 7, 7, 5, 5, 5, 7, 7, 5, 3]


# ================================
# 1.3 Analyse statistique
# ================================

# Calcul du nombre total de données (taille de l'échantillon)
n = len(DATA)

# Calcul de la moyenne empirique du temps de sommeil
moy = np.mean(DATA)

# Calcul de la variance empirique corrigée
# ddof=1 permet d'utiliser la formule adaptée à un échantillon
var = np.var(DATA, ddof=1)

# Calcul de l'écart-type à partir de la variance
std = np.sqrt(var)

# Calcul de la médiane des données
med = np.median(DATA)

# Recherche de la valeur minimale et maximale
mini, maxi = np.min(DATA), np.max(DATA)


# ================================
# Affichage des résultats
# ================================

print("--- PARTIE I : ANALYSE STATISTIQUE DU SOMMEIL ---")

# Affichage du nombre de réponses du sondage
print(f"1. Taille échantillon : {n}")

# Affichage de la moyenne avec 2 chiffres après la virgule
print(f"2. Moyenne empirique : {moy:.2f}h")

# Affichage de la variance avec 4 chiffres après la virgule
print(f"3. Variance empirique : {var:.4f}")

# Affichage de l'écart-type avec 4 chiffres après la virgule
print(f"4. Écart-type empirique : {std:.4f}")

# Affichage de la médiane
print(f"5. Médiane : {med:.2f}h")

# Affichage des valeurs extrêmes
print(f"6. Minimum : {mini}h | 7. Maximum : {maxi}h")

# Création d'une figure avec 3 sous-graphiques (1 ligne, 3 colonnes)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Bornes des classes (IMPORTANT pour respecter les intervalles)
bins = [2, 4, 6, 8, 10]

# ================================
# 1. Histogramme
# ================================
axes[0].hist(DATA, bins=bins, edgecolor="black")
axes[0].set_title("Histogramme des heures de sommeil")
axes[0].set_xlabel("Heures de sommeil")
axes[0].set_ylabel("Effectif")
axes[0].set_xticks([3, 5, 7, 9])

# ================================
# 2. Boîte à moustaches
# ================================
axes[1].boxplot(DATA, vert=False)
axes[1].set_title("Boîte à moustaches")
axes[1].set_xlabel("Heures de sommeil")


# ================================
# 3. Courbe de densité (approximation)
# ================================
kde = gaussian_kde(DATA)  # estimation de densité

x = np.linspace(min(DATA), max(DATA), 200)
axes[2].plot(x, kde(x))
axes[2].fill_between(x, kde(x), alpha=0.3)

axes[2].set_title("Courbe de densité")
axes[2].set_xlabel("Heures de sommeil")
axes[2].set_ylabel("Densité")


# Ajustement automatique de l'espacement
plt.tight_layout()

# Affichage
plt.show()