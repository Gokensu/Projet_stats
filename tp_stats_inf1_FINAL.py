import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# PARTIE I : ÉTUDE D'UN PHÉNOMÈNE RÉEL (Sommeil)
# ============================================================
# 1.2 Collecte des données (Échantillon n=18 extrait de tes images)
DATA = [3.5, 4.5, 4.5, 5, 5, 5.5, 5.5, 6, 6, 6, 6.5, 6.5, 7, 7, 7.5, 7.5, 8, 8]

# 1.3 Analyse statistique avec Python
n = len(DATA)
moy = np.mean(DATA)
var = np.var(DATA, ddof=1)
std = np.std(DATA, ddof=1)
med = np.median(DATA)
mini, maxi = np.min(DATA), np.max(DATA)

print("--- PARTIE I : ANALYSE STATISTIQUE DU SOMMEIL ---")
print(f"1. Taille échantillon : {n}")
print(f"2. Moyenne empirique : {moy:.2f}h")
print(f"3. Variance empirique : {var:.4f}")
print(f"4. Écart-type empirique : {std:.4f}")
print(f"5. Médiane : {med:.2f}h")
print(f"6. Minimum : {mini}h | 7. Maximum : {maxi}h")

# 1.5 Modélisation probabiliste
# On choisit la loi normale car la variable est continue et biologique.
loi_normale = stats.norm(loc=moy, scale=std)

# 1.6 Étude de la loi (Calculs)
print(f"\n--- MODÉLISATION (Loi Normale) ---")
print(f"1. P(X ≤ 6h) : {loi_normale.cdf(6):.4f}")
print(f"   P(X > 7h) : {1 - loi_normale.cdf(7):.4f}")
print(f"   P(5h < X < 8h) : {loi_normale.cdf(8) - loi_normale.cdf(5):.4f}")
print(f"2. F(5h) = P(X ≤ 5h) : {loi_normale.cdf(5):.4f}")
print(f"3. 5 valeurs aléatoires générées : {loi_normale.rvs(size=5)}")

# 1.4 & 1.6 Représentations graphiques
x_range = np.linspace(2, 10, 100)
plt.figure(figsize=(15, 5))

# Histogramme et Densité
plt.subplot(1, 3, 1)
plt.hist(DATA, bins=6, density=True, alpha=0.6, color='skyblue', label="Données")
plt.plot(x_range, loi_normale.pdf(x_range), 'r-', lw=2, label="Densité PDF")
plt.title("1.4 & 1.6 : Histogramme et Densité")
plt.legend()

# Boîte à moustaches
plt.subplot(1, 3, 2)
plt.boxplot(DATA, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
plt.title("1.4 : Boîte à moustaches")

# Fonction de répartition
plt.subplot(1, 3, 3)
plt.plot(x_range, loi_normale.cdf(x_range), 'b-', lw=2)
plt.title("1.6 : Fonction de répartition (CDF)")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# PARTIE II : ÉTUDE DES LOIS CONTINUES
# ============================================================
print("\n" + "="*50)
print("PARTIE II : ÉTUDE DES LOIS CONTINUES")
print("="*50)

# 2.1 Travail demandé pour chaque loi (Définitions et Commandes)
def afficher_aide_memoire():
    print("COMMANDES PYTHON (scipy.stats) :")
    print("- Densité f(x) : .pdf(x)")
    print("- Répartition F(x) : .cdf(x)")
    print("- Quantiles : .ppf(q)")
    print("- Aléatoire : .rvs(size)")

afficher_aide_memoire()

# Calculs de probabilités (Exemples)
def calculs_intervalles(nom_loi, loi_objet, a, b):
    print(f"\nCalculs pour {nom_loi}:")
    print(f"  P(X ≤ {a})    : {loi_objet.cdf(a):.4f}")
    print(f"  P(X > {a})    : {1 - loi_objet.cdf(a):.4f}")
    print(f"  P({a} < X < {b}) : {loi_objet.cdf(b) - loi_objet.cdf(a):.4f}")

calculs_intervalles("Student (v=10)", stats.t(df=10), a=1.5, b=2.5)
calculs_intervalles("Khi-deux (k=5)", stats.chi2(df=5), a=2, b=8)

# 2.2 Comparaisons graphiques
plt.figure(figsize=(15, 10))

# 1. Normale vs Student (Étude de l'effet des degrés de liberté)
# Plus v est petit, plus les "queues" sont épaisses. Quand v -> inf, Student -> Normale.

plt.subplot(2, 2, 1)
x_std = np.linspace(-4, 4, 500)
plt.plot(x_std, stats.norm.pdf(x_std), 'k--', lw=2, label="Normale (0,1)")
for df in [1, 2, 30]:
    plt.plot(x_std, stats.t.pdf(x_std, df), label=f"Student (v={df})")
plt.title("2.2 : Loi Normale vs Student")
plt.legend()

# 2. Plusieurs lois du Khi-deux (Étude de l'effet de k)
# La loi est asymétrique. Plus k augmente, plus elle se déplace vers la droite et devient symétrique.

plt.subplot(2, 2, 2)
x_chi = np.linspace(0, 20, 500)
for k in [2, 5, 10]:
    plt.plot(x_chi, stats.chi2.pdf(x_chi, k), label=f"Khi-deux (k={k})")
plt.title("2.2 : Lois du Khi-deux")
plt.legend()

# 3. Plusieurs lois de Fisher (Étude de d1, d2)
plt.subplot(2, 2, 3)
x_f = np.linspace(0, 5, 500)
for d1, d2 in [(2,10), (5,20), (10,30)]:
    plt.plot(x_f, stats.f.pdf(x_f, d1, d2), label=f"Fisher (d1={d1}, d2={d2})")
plt.title("2.2 : Lois de Fisher")
plt.legend()

# 4. Fonction de répartition F(x) et Interprétation
# F(x) donne la probabilité cumulée. F(médiane) = 0.5.
plt.subplot(2, 2, 4)
plt.plot(x_std, stats.norm.cdf(x_std), 'k--', label="CDF Normale")
plt.plot(x_std, stats.t.cdf(x_std, df=2), label="CDF Student (v=2)")
plt.axhline(0.5, color='gray', ls=':', alpha=0.5)
plt.title("2.3 : Fonctions de répartition")
plt.legend()

plt.tight_layout()
plt.show()

print("\n✓ Script terminé. Les graphiques et les calculs sont prêts pour le compte rendu.")