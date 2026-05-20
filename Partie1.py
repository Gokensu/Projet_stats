# Importation de la bibliothèque NumPy, standard pour les calculs mathématiques et la gestion des tableaux numériques. Assignée à l'alias 'np'.
import numpy as np

# Importation du module pyplot de Matplotlib, utilisé pour générer des graphiques et des visualisations. Assigné à l'alias 'plt'.
import matplotlib.pyplot as plt

# Importation du module stats de SciPy, qui contient des fonctions statistiques avancées (utilisé ici pour l'estimation de densité).
from scipy import stats

# Importation de Pandas, une bibliothèque conçue pour la manipulation et l'analyse de données structurées. Assignée à l'alias 'pd'.
import pandas as pd

# ============================================================
# __Données__ intervalles
# ============================================================

# Chargement du fichier CSV dans un "DataFrame" Pandas (une structure de données en tableau). La variable 'df' stocke ce tableau.
df = pd.read_csv("../csv/sondage.csv")

# Renommage forcé des colonnes du DataFrame 'df' pour faciliter l'accès aux données.
df.columns = ["horodateur", "td", "heures", "raisons"]

# Création d'un dictionnaire nommé 'sondage' servant de table de conversion.
# Les clés sont les réponses textuelles, les valeurs sont les points milieux numériques.
sondage = {
    "Entre 6h et 8h": 7,
    "Entre 4h et 6h": 5,
    "Moins de 4h": 3,
}

# Initialisation d'une liste vide pour stocker les valeurs numériques après conversion.
heures_converties = []

# Boucle itérative pour analyser chaque cellule de la colonne "heures".
for reponse in df["heures"]:
    
    # Vérification si la chaîne de caractères existe en tant que clé dans le dictionnaire.
    if reponse in sondage:
        
        # Extraction de la valeur numérique correspondante.
        val = sondage[reponse]
        
        # Ajout de cette valeur numérique à la fin de la liste.
        heures_converties.append(val)

# Assignation de la liste finale à la variable principale 'heures'.
heures = heures_converties

# Utilisation de la fonction len() pour compter le nombre total d'éléments (taille de l'échantillon).
n = len(heures)

# ============================================================
# ANALYSE STATISTIQUE
# ============================================================

# Calcul de la moyenne arithmétique via NumPy.
moyenne = np.mean(heures)

# Calcul de la variance. Le paramètre 'ddof=1' indique une variance empirique non biaisée.
variance = np.var(heures, ddof=1)

# Calcul de l'écart-type (racine carrée de la variance).
ecart_type = np.sqrt(variance)

# Identification de la médiane (la valeur séparant l'échantillon en deux moitiés égales).
mediane = np.median(heures)

# Identification de la valeur la plus basse de l'échantillon.
minimum = np.min(heures)

# Identification de la valeur la plus haute de l'échantillon.
maximum = np.max(heures)

# ============================================================
# BLOC D'EXÉCUTION
# Si ce fichier est importé par 'partie2.py', ce bloc sera ignoré pour ne pas afficher les graphiques prématurément.
if __name__ == "__main__":
    
    # Impression des résultats dans la console avec formatage à deux décimales (:.2f).
    print("--- PARTIE I : ANALYSE STATISTIQUE DU SOMMEIL ---")
    print(f"Taille de l'échantillon : {n}")
    print(f"Moyenne : {moyenne:.2f} heures")
    print(f"Variance : {variance:.2f} heures^2")
    print(f"Ecart-type : {ecart_type:.2f} heures")
    print(f"Médiane : {mediane:.2f} heures")
    print(f"Minimum : {minimum:.2f} heures")
    print(f"Maximum : {maximum:.2f} heures")

    # 1. Histogramme
    # Création d'un histogramme avec des limites de classes définies par 'bins'.
    plt.hist(heures, bins=[2, 4, 6, 8], edgecolor='black')
    plt.title("Distribution des heures de sommeil")
    plt.xlabel("Heures de sommeil")
    plt.ylabel("Nombre de personnes")
    plt.xticks([2, 3, 4, 5, 6, 7, 8]) # Force l'affichage de l'axe X
    plt.show() # Affichage de la fenêtre graphique

    # 2. Boîte à moustaches
    # Création d'un diagramme en boîte horizontal (vert=False).
    plt.boxplot(heures, vert=False)
    plt.title("Boîte à moustaches des heures de sommeil")
    plt.xlabel("Heures de sommeil")
    plt.yticks([1], ['']) # Masque l'étiquette inutile sur l'axe Y
    plt.show()

    # 3. Courbe de densité
    # Création d'un tableau continu pour l'axe des abscisses.
    x = np.linspace(2, 10, 100)
    # Modèle d'estimation de densité par noyau basé sur les données de l'échantillon.
    kde = stats.gaussian_kde(heures)
    
    # Tracé de la courbe.
    plt.plot(x, kde(x), label='Densité de probabilité')
    plt.title("Courbe de densité des heures de sommeil")
    plt.xlabel("Heures de sommeil")
    plt.ylabel("Densité")
    plt.legend()
    plt.show()