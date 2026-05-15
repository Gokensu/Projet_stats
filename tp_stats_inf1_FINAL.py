import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- CONFIGURATION ---
DATA = [6.5, 7.0, 5.5, 8.0, 6.0, 7.5, 5.0, 6.5, 7.0, 6.0, 5.5, 8.5, 6.0, 7.0, 5.5, 
        6.0, 7.5, 6.5, 5.0, 7.0, 6.0, 5.5, 7.0, 6.5, 8.0, 5.5, 6.0, 7.0, 6.5, 5.5]
NOM, UNITE = "Sommeil CPGE", "heures"

# --- 1. ANALYSE STATISTIQUE ---
moy, std = np.mean(DATA), np.std(DATA, ddof=1)
print(f"Stats: Moyenne = {moy:.2f}{UNITE} | Écart-type = {std:.2f}")

# Test de normalité (Shapiro-Wilk)
_, p_val = stats.shapiro(DATA)
print(f"Normalité : {'OUI' if p_val > 0.05 else 'NON'} (p={p_val:.4f})")

# --- 2. MODÉLISATION (Loi Normale) ---
# On ajuste une loi normale aux données
loi = stats.norm(loc=moy, scale=std)
x = np.linspace(min(DATA)-1, max(DATA)+1, 100)

# --- 3. GRAPHIQUE UNIQUE ---
plt.figure(figsize=(10, 5))

# Histogramme des données
plt.hist(DATA, bins=8, density=True, alpha=0.6, color='skyblue', label="Données")

# Courbe de la loi normale ajustée
plt.plot(x, loi.pdf(x), 'r-', lw=2, label="Modèle (Loi Normale)")

plt.title(f"Analyse du {NOM}")
plt.xlabel(UNITE)
plt.ylabel("Densité")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# --- 4. CALCULS RAPIDES ---
print(f"\nProbabilité de dormir moins de 6h : {loi.cdf(6):.2%}")
print(f"Intervalle de confiance (95%) : [{loi.ppf(0.025):.2f} ; {loi.ppf(0.975):.2f}] {UNITE}")