# Importation de la bibliothèque NumPy, standard pour les calculs mathématiques et la gestion des tableaux numériques. Assignée à l'alias 'np'.
import numpy as np

# Importation du module pyplot de Matplotlib, utilisé pour générer des graphiques et des visualisations. Assigné à l'alias 'plt'.
import matplotlib.pyplot as plt

# Importation du module stats de SciPy, qui contient des fonctions statistiques avancées (utilisé ici pour l'estimation de densité).
from scipy import stats

# Importation de Pandas, une bibliothèque conçue pour la manipulation et l'analyse de données structurées (comme les fichiers CSV). Assignée à l'alias 'pd'.
import pandas as pd

# ============================================================
# __Données__ intervalles
# ============================================================

# Chargement du fichierCSV dans un "DataFrame" Pandas (une structure de données en tableau). La variable 'df' stocke ce tableau.
df = pd.read_csv("../csv/sondage.csv")

# Renommage forcé des colonnes du DataFrame 'df' pour faciliter l'accès aux données. Nous avons 4 colonnes correspondant aux réponses du formulaire.
df.columns = ["horodateur", "td", "heures", "raisons"]

# Création d'un dictionnaire nommé 'sondage'. Il sert de table de conversion.
# Les clés (à gauche) sont les réponses textuelles attendues, les valeurs (à droite) sont les points milieux numériques associés.
sondage = {
    "Entre 6h et 8h": 7,
    "Entre 4h et 6h": 5,
    "Moins de 4h": 3,
}

# Initialisation d'une liste vide nommée 'heures_converties'. Elle servira de réceptacle pour stocker les valeurs numériques après conversion.
heures_converties = []

# Début d'une boucle itérative. La variable 'reponse' va prendre tour à tour la valeur de chaque cellule de la colonne "heures" du tableau 'df'.
for reponse in df["heures"]:
    
    # Structure conditionnelle (if) pour vérifier si la chaîne de caractères contenue dans 'reponse' existe en tant que clé dans notre dictionnaire 'sondage'.
    if reponse in sondage:
        
        # Si la condition est vraie, on extrait la valeur numérique correspondante du dictionnaire et on la stocke temporairement dans la variable 'val'.
        val = sondage[reponse]
        
        # On ajoute cette valeur numérique ('val') à la fin de notre liste 'heures_converties'.
        heures_converties.append(val)

# Assignation de la liste finale à la variable principale 'heures'. Cette variable devient notre jeu de données de référence pour tout le reste du script.
heures = heures_converties

# Utilisation de la fonction len() pour compter le nombre total d'éléments dans la liste 'heures'. Le résultat est stocké dans la variable 'n', qui représente la taille de l'échantillon.
n = len(heures)

# ============================================================
# ANALYSE STATISTIQUE
# ============================================================

# Calcul de la moyenne arithmétique de la liste 'heures' via NumPy. Le résultat est assigné à la variable 'moyenne'.
moyenne = np.mean(heures)

# Calcul de la variance. Le paramètre 'ddof=1' (Delta Degrees of Freedom) indique qu'il s'agit d'une variance empirique non biaisée (divisée par n-1 au lieu de n). Assignée à 'variance'.
variance = np.var(heures, ddof=1)

# Calcul de l'écart-type, qui est mathématiquement la racine carrée (sqrt pour square root) de la variance. Assigné à 'ecart_type'.
ecart_type = np.sqrt(variance)

# Identification de la médiane (la valeur qui sépare l'échantillon en deux moitiés égales) via NumPy. Assignée à 'mediane'.
mediane = np.median(heures)

# Identification de la valeur la plus basse de l'échantillon. Assignée à 'minimum'.
minimum = np.min(heures)

# Identification de la valeur la plus haute de l'échantillon. Assignée à 'maximum'.
maximum = np.max(heures)

# ============================================================
# AFFICHAGE DES RÉSULTATS TEXTUELS
# ============================================================

# Impression des résultats dans la console. Le 'f' devant les guillemets indique une chaîne formatée (f-string), permettant d'insérer des variables directement entre accolades {}.
print(f"Taille de l'échantillon : {n}")
# Le format ':.2f' indique que la variable numérique doit être affichée avec une précision de 2 décimales (chiffres après la virgule).
print(f"Moyenne : {moyenne:.2f} heures")
print(f"Variance : {variance:.2f} heures^2")
print(f"Ecart-type : {ecart_type:.2f} heures")
print(f"Médiane : {mediane:.2f} heures")
print(f"Minimum : {minimum:.2f} heures")
print(f"Maximum : {maximum:.2f} heures")

# ============================================================
# REPRÉSENTATIONS GRAPHIQUES
# ============================================================

# 1. Histogramme
# Création d'un histogramme basé sur la liste 'heures'. Le paramètre 'bins' définit explicitement les limites des classes (colonnes) pour isoler les valeurs 3, 5 et 7. 'edgecolor' trace le contour des barres.
plt.hist(heures, bins=[2, 4, 6, 8], edgecolor='black')
plt.title("Distribution des heures de sommeil") # Définit le titre du graphique
plt.xlabel("Heures de sommeil") # Définit le label de l'axe des abscisses (X)
plt.ylabel("Nombre de personnes") # Définit le label de l'axe des ordonnées (Y)
plt.xticks([2, 3, 4, 5, 6, 7, 8]) # Force l'affichage de valeurs spécifiques sur l'axe des X pour une lecture claire.
plt.show() # Commande d'exécution qui génère et affiche la fenêtre graphique.

# 2. Boîte à moustaches
# Création d'un diagramme en boîte à moustaches (boxplot). Le paramètre 'vert=False' indique que le graphique doit être dessiné horizontalement.
plt.boxplot(heures, vert=False)
plt.title("Boîte à moustaches des heures de sommeil")
plt.xlabel("Heures de sommeil")
plt.yticks([1], ['']) # Masque l'étiquette par défaut de l'axe Y (qui afficherait un '1' inutile) en la remplaçant par une chaîne vide.
plt.show()

# 3. Courbe de densité
# Création d'un tableau 'x' contenant 100 valeurs espacées uniformément entre 2 et 10. Cela sert d'axe d'abscisses continu pour tracer une courbe fluide.
x = np.linspace(2, 10, 100)

# Initialisation d'un modèle d'estimation de densité par noyau (KDE - Kernel Density Estimation) fourni par SciPy, calibré sur les données de 'heures'. La fonction résultante est assignée à 'kde'.
kde = stats.gaussian_kde(heures)

# Tracé d'une courbe en ligne (plot). L'axe X est le tableau 'x', l'axe Y est généré en évaluant la fonction 'kde' sur chaque point de 'x'. L'argument 'label' sert pour la légende.
plt.plot(x, kde(x), label='Densité de probabilité')
plt.title("Courbe de densité des heures de sommeil")
plt.xlabel("Heures de sommeil")
plt.ylabel("Densité")
plt.legend() # Active l'affichage de la légende définie précédemment dans la commande plot.
plt.show()