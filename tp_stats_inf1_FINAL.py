import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
#  CONFIGURATION
# ============================================================
PHENOMENE  = "Temps de sommeil des étudiants CPGE"
UNITE      = "heures"
DONNEES    = [6.5, 7.0, 5.5, 8.0, 6.0, 7.5, 5.0, 6.5, 7.0, 6.0,
              5.5, 8.5, 6.0, 7.0, 5.5, 6.0, 7.5, 6.5, 5.0, 7.0,
              6.0, 5.5, 7.0, 6.5, 8.0, 5.5, 6.0, 7.0, 6.5, 5.5]
LOI_CHOISIE = "norm" # "norm", "expon", ou "uniform"

# ============================================================
#  FONCTIONS OUTILS
# ============================================================

def afficher_stats(data):
    """Calcule et affiche les statistiques descriptives."""
    s = stats.describe(data)
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    print(f"\n--- STATISTIQUES ({PHENOMENE}) ---")
    print(f"n: {s.nobs} | Moyenne: {s.mean:.3f} | Écart-type: {np.sqrt(s.variance):.3f}")
    print(f"Médiane: {med:.3f} | Q1: {q1:.3f} | Q3: {q3:.3f} | IQR: {q3-q1:.3f}")
    
    sw_stat, sw_p = stats.shapiro(data)
    print(f"Test Shapiro-Wilk: p-value = {sw_p:.4f}")
    print(f"-> Normalité {'acceptée' if sw_p > 0.05 else 'rejetée'} (seuil 5%)")

def tracer_distribution(data, loi_nom):
    """Affiche l'analyse graphique et le fit de la loi choisie."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Histogramme + KDE
    axes[0].hist(data, bins='auto', density=True, alpha=0.5, color='steelblue', label='Données')
    kde_x = np.linspace(min(data)-1, max(data)+1, 100)
    axes[0].plot(kde_x, stats.gaussian_kde(data)(kde_x), 'r--', label='KDE')
    
    # 2. Ajustement de la loi
    dist = getattr(stats, loi_nom)
    params = dist.fit(data)
    axes[0].plot(kde_x, dist.pdf(kde_x, *params), 'k-', lw=2, label=f'Fit {loi_nom}')
    axes[0].set_title("Distribution et Ajustement")
    axes[0].legend()

    # 3. Boxplot
    axes[1].boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    axes[1].set_title("Boîte à moustaches")
    
    plt.tight_layout()
    plt.show()
    return dist(*params)

# ============================================================
#  EXÉCUTION DU TP
# ============================================================

# Partie I : Analyse des données réelles
donnees = np.array(DONNEES)
afficher_stats(donnees)
loi_fit = tracer_distribution(donnees, LOI_CHOISIE)

# Calculs de probabilités sur le modèle ajusté
print(f"\n--- MODÉLISATION ({LOI_CHOISIE}) ---")
a, b = np.mean(donnees) - np.std(donnees), np.mean(donnees) + np.std(donnees)
print(f"P({a:.1f} < X < {b:.1f}) = {loi_fit.cdf(b) - loi_fit.cdf(a):.4f}")
print(f"Intervalle à 95% (quantiles) : [{loi_fit.ppf(0.025):.2f} ; {loi_fit.ppf(0.975):.2f}]")

# Partie II : Comparaison théorique (Focus Student vs Normale)
print("\n--- ANALYSE THÉORIQUE (Lois usuelles) ---")
x = np.linspace(-4, 4, 200)
plt.figure(figsize=(8, 4))
plt.plot(x, stats.norm.pdf(x), 'k-', lw=2, label='N(0,1)')
for ddl in [1, 5, 30]:
    plt.plot(x, stats.t.pdf(x, df=ddl), '--', label=f'Student (ddl={ddl})')
plt.title("Convergence de la loi de Student vers la loi Normale")
plt.legend()
plt.grid(alpha=0.3)
plt.show()