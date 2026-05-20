# Importation des bibliothèques mathématiques et graphiques nécessaires.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# LIAISON AVEC LE FICHIER 1
# ============================================================
from Partie1 import n, moyenne, variance, ecart_type, mediane, heures

# ============================================================
# PARTIE II : ÉTUDE DES LOIS CONTINUES
# ============================================================

# 2.1 Travail demandé pour chaque loi (Définitions et Commandes)

# Calculs de probabilités (Exemples)
def calculs_intervalles(nom_loi, loi_objet, a, b):
    # Fonction automatisant le calcul des probabilités. Reçoit le nom, l'objet statistique, et les bornes a et b.
    print(f"\nCalculs pour {nom_loi}:")
    # P(X ≤ a) : Fonction de répartition (CDF)
    print(f"  P(X ≤ {a})    : {loi_objet.cdf(a):.4f}")
    # P(X > a) : Complémentaire de la fonction de répartition
    print(f"  P(X > {a})    : {1 - loi_objet.cdf(a):.4f}")
    # P(a < X < b) : Différence entre les deux bornes sur la CDF
    print(f"  P({a} < X < {b}) : {loi_objet.cdf(b) - loi_objet.cdf(a):.4f}")

# Application de la fonction de calcul sur une loi de Student (10 degrés de liberté).
calculs_intervalles("Student (v=10)", stats.t(df=10), a=1.5, b=2.5)

# Application de la fonction de calcul sur une loi du Khi-deux (5 degrés de liberté).
calculs_intervalles("Khi-deux (k=5)", stats.chi2(df=5), a=2, b=8)

# ============================================================
# 2.2 Comparaisons graphiques
# ============================================================
# Initialisation de la fenêtre graphique principale (15 pouces de large, 10 pouces de haut).
plt.figure(figsize=(15, 10))

# ------------------------------------------------------------
# 1. Normale vs Student
# Création du premier sous-graphique (grille 2x2, position 1).
plt.subplot(2, 2, 1)
# Définition de l'axe des abscisses centré sur 0.
x_std = np.linspace(-4, 4, 500)
# Tracé de la loi normale standard (ligne noire pointillée).
plt.plot(x_std, stats.norm.pdf(x_std), 'k--', lw=2, label="Normale (0,1)")

# Boucle pour générer la loi de Student avec 1, 2, puis 30 degrés de liberté.
for deg_lib in [1, 2, 30]:
    plt.plot(x_std, stats.t.pdf(x_std, deg_lib), label=f"Student (v={deg_lib})")
plt.title("2.2 : Loi Normale vs Student")
plt.legend()

# ------------------------------------------------------------
# 2. Plusieurs lois du Khi-deux
# Création du deuxième sous-graphique (grille 2x2, position 2).
plt.subplot(2, 2, 2)
# Définition de l'axe des abscisses (strictement positif pour le Khi-deux).
x_chi = np.linspace(0, 20, 500)

# Boucle pour générer la loi du Khi-deux avec 2, 5, puis 10 degrés de liberté.
for k in [2, 5, 10]:
    plt.plot(x_chi, stats.chi2.pdf(x_chi, k), label=f"Khi-deux (k={k})")
plt.title("2.2 : Lois du Khi-deux")
plt.legend()

# ------------------------------------------------------------
# 3. Plusieurs lois de Fisher
# Création du troisième sous-graphique (grille 2x2, position 3).
plt.subplot(2, 2, 3)
x_f = np.linspace(0, 5, 500)

# Boucle itérant sur des paires de degrés de liberté (numérateur, dénominateur).
for d1, d2 in [(2,10), (5,20), (10,30)]:
    plt.plot(x_f, stats.f.pdf(x_f, d1, d2), label=f"Fisher (d1={d1}, d2={d2})")
plt.title("2.2 : Lois de Fisher")
plt.legend()

# ------------------------------------------------------------
# 4. Fonctions de répartition
# Création du quatrième sous-graphique (grille 2x2, position 4).
plt.subplot(2, 2, 4)
# Tracé de la CDF de la loi normale standard.
plt.plot(x_std, stats.norm.cdf(x_std), 'k--', label="CDF Normale")
# Tracé de la CDF de la loi de Student (2 degrés de liberté).
plt.plot(x_std, stats.t.cdf(x_std, df=2), label="CDF Student (v=2)")
# Ajout d'une ligne de repère à 50% (médiane).
plt.axhline(0.5, color='gray', ls=':', alpha=0.5)
plt.title("2.3 : Fonctions de répartition")
plt.legend()

# ------------------------------------------------------------
# Ajustement de l'espacement pour éviter les chevauchements de texte entre les graphiques.
plt.tight_layout()
# Commande finale pour générer la fenêtre visuelle.
plt.show()

print("\n✓ Script terminé. Les graphiques et les calculs sont prêts pour le compte rendu.")