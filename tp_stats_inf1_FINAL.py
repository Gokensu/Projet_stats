import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# PARTIE I : ÉTUDE D'UN PHÉNOMÈNE RÉEL (Sommeil)
# ============================================================
# Données extraites du sondage (n=18) [cite: 23, 31]
DATA = [3.5, 4.5, 4.5, 5, 5, 5.5, 5.5, 6, 6, 6, 6.5, 6.5, 7, 7, 7.5, 7.5, 8, 8]
moy, std = np.mean(DATA), np.std(DATA, ddof=1)
loi_normale = stats.norm(loc=moy, scale=std)

print("--- PARTIE I : ANALYSE DU SOMMEIL ---")

# 1. Calculer des probabilités [cite: 52]
print(f"1. Probabilité P(X ≤ 6h) : {loi_normale.cdf(6):.4f}")
print(f"   Probabilité P(X > 7h) : {1 - loi_normale.cdf(7):.4f}")

# 2. Calculer la fonction de répartition en un point [cite: 53]
print(f"2. F(5h) = P(X ≤ 5h) : {loi_normale.cdf(5):.4f}")

# 3. Générer des valeurs aléatoires [cite: 54]
print(f"3. 10 valeurs simulées : {loi_normale.rvs(size=10)}")

# 4 & 5. Tracer Densité et Répartition [cite: 55, 56]
x_range = np.linspace(2, 10, 100)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(x_range, loi_normale.pdf(x_range), 'r-')
plt.title("Densité de probabilité (PDF)")
plt.subplot(1, 2, 2)
plt.plot(x_range, loi_normale.cdf(x_range), 'b-')
plt.title("Fonction de répartition (CDF)")
plt.show()

# ============================================================
# PARTIE II : ÉTUDE DES LOIS CONTINUES [cite: 64, 70]
# ============================================================
print("\n--- PARTIE II : ÉTUDE DES LOIS CONTINUES ---")

def analyser_loi(nom, loi, a, b):
    """Calcule les probabilités demandées pour une loi [cite: 83-86]"""
    p_inf = loi.cdf(a)
    p_sup = 1 - loi.cdf(a)
    p_inter = loi.cdf(b) - loi.cdf(a)
    print(f"Loi {nom} -> P(X≤{a}): {p_inf:.4f} | P(X>{a}): {p_sup:.4f} | P({a}<X<{b}): {p_inter:.4f}")

# Exemples de calculs [cite: 87]
analyser_loi("Student (v=5)", stats.t(df=5), a=2, b=3)
analyser_loi("Khi-deux (k=10)", stats.chi2(df=10), a=5, b=15)

# --- COMPARAISONS GRAPHIQUES [cite: 98] ---
x = np.linspace(-5, 5, 500)
plt.figure(figsize=(15, 8))

# 1. Loi normale vs Student [cite: 100]
plt.subplot(2, 2, 1)
plt.plot(x, stats.norm.pdf(x), 'k--', label="N(0,1)")
for df in [1, 5, 30]: # Étude de l'effet des ddl [cite: 97, 101]
    plt.plot(x, stats.t.pdf(x, df), label=f"Student v={df}")
plt.title("Normale vs Student")
plt.legend()

# 2. Plusieurs lois du Khi-deux [cite: 102]
plt.subplot(2, 2, 2)
x_c = np.linspace(0, 20, 500)
for k in [2, 5, 10]:
    plt.plot(x_c, stats.chi2.pdf(x_c, k), label=f"Khi2 k={k}")
plt.title("Plusieurs lois du Khi-deux")
plt.legend()

# 3. Plusieurs lois de Fisher [cite: 103]
plt.subplot(2, 2, 3)
x_f = np.linspace(0, 5, 500)
for d1, d2 in [(2,10), (5,20), (10,30)]:
    plt.plot(x_f, stats.f.pdf(x_f, d1, d2), label=f"F({d1},{d2})")
plt.title("Plusieurs lois de Fisher")
plt.legend()

# 4. Fonctions de répartition F(x) [cite: 104, 107]
plt.subplot(2, 2, 4)
plt.plot(x, stats.norm.cdf(x), label="F(x) Normale")
plt.plot(x, stats.t.cdf(x, df=5), label="F(x) Student v=5")
plt.title("Fonctions de répartition")
plt.legend()

plt.tight_layout()
plt.show()