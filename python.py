import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
"""
#__Données__ intervalles

heures = [3]*1 +[5]*8 + [7]*9  # 8 personnes entre 4h et 6h, 9 personnes entre 6h et 8h

# ============================================================
#Taille de l'échantillon
n=len(heures)

#Moyenne intervalles
moyenne=np.mean(heures)

#variance intervalles
variance=np.var(heures)

#Ecart-type
ecart_type=np.sqrt(variance)

#médiane
mediane=np.median(heures)

#minimum
minimum=np.min(heures)

#maximum
maximum=np.max(heures)

#affichage des résultats
print(f"Taille de l'échantillon : {n}")
print(f"Moyenne : {moyenne:.2f} heures")
print(f"Variance : {variance:.2f} heures^2")
print(f"Ecart-type : {ecart_type:.2f} heures")
print(f"Médiane : {mediane:.2f} heures")
print(f"Minimum : {minimum:.2f} heures")
print(f"Maximum : {maximum:.2f} heures")

#representation graphe sur le meme graphique (3graphes sur une image)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Analyse du sommeil en semaine — Classe (n=18)", fontsize=13)

axes[0].hist(heures, bins=[4, 6, 8], edgecolor='black')
axes[0].set_title("Distribution des heures de sommeil")
axes[0].set_xlabel("Heures de sommeil")
axes[0].set_ylabel("Nombre de personnes")
axes[0].set_xticks([4, 5, 6, 7, 8])
plt.show()

#boite à moustaches
axes[1].boxplot(heures, vert=False)
axes[1].set_title("Boîte à moustaches des heures de sommeil")
axes[1].set_xlabel("Heures de sommeil")
axes[1].set_yticks([1], [''])
axes[1].show()

#courbe de densité

x = np.linspace(4, 8, 100)
kde = stats.gaussian_kde(heures)

axes[2].plot(x, kde(x), label='Densité de probabilité')
axes[2].set_title("Courbe de densité des heures de sommeil")
axes[2].set_xlabel("Heures de sommeil")
axes[2].set_ylabel("Densité")
axes[2].legend()
plt.show()

"""

# ============================================================
# Données réelles — 3 classes
# Moins de 4h → milieu = 3  (n=1)
# Entre 4h-6h → milieu = 5  (n=11)
# Entre 6h-8h → milieu = 7  (n=6)

heures = [3]*1 + [5]*11 + [7]*6

# ============================================================
# 1. Taille de l'échantillon
n = len(heures)

# 2. Fréquences
milieux    = np.array([3, 5, 7])
effectifs  = np.array([1, 11, 6])
frequences = effectifs / n          # fᵢ = nᵢ / n

# 3. Moyenne  →  x̄ = Σ fᵢ · xᵢ
moyenne = np.sum(frequences * milieux)

# 4. Variance  →  σ² = Σ fᵢ · xᵢ² − x̄²
variance = np.sum(frequences * milieux**2) - moyenne**2

# 5. Écart-type  →  σ = √σ²
ecart_type = np.sqrt(variance)

# 6. Médiane par interpolation linéaire
#    Fréquences cumulées : F1=1/18, F2=12/18, F3=18/18
#    0.5 tombe dans [4h, 6h]  →  a=4, h=2, F_avant=1/18, f_med=11/18
a       = 4
h       = 2
F_avant = 1 / n
f_med   = 11 / n
mediane = a + h * (0.5 - F_avant) / f_med

# 7. Min / Max (bornes des classes extrêmes)
minimum = 0   # borne inférieure de "Moins de 4h"
maximum = 8   # borne supérieure de "Entre 6h-8h"

# ============================================================
# Affichage
print(f"Taille de l'échantillon : {n}")
print(f"Moyenne                 : {moyenne:.4f} heures")
print(f"Variance                : {variance:.4f} heures²")
print(f"Écart-type              : {ecart_type:.4f} heures")
print(f"Médiane                 : {mediane:.4f} heures")
print(f"Minimum (borne)         : {minimum} heures")
print(f"Maximum (borne)         : {maximum} heures")

# ============================================================
# Graphiques
labels   = ["< 4h", "4h–6h", "6h–8h"]
freq_pct = frequences * 100   # en pourcentage

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Analyse du sommeil en semaine — Classe (n=18)", fontsize=13)

# --- Graphe 1 : Histogramme des effectifs ---
axes[0].bar(labels, effectifs, color=["#AFA9EC", "#7F77DD", "#534AB7"],
            edgecolor="white", width=0.6)
axes[0].set_title("Effectifs par classe")
axes[0].set_xlabel("Heures de sommeil")
axes[0].set_ylabel("Nombre d'étudiants")
axes[0].axvline(x=1, color="#D85A30", linestyle="--", linewidth=1.2,
                label=f"Médiane ≈ {mediane:.2f}h")
axes[0].legend(fontsize=9)
for i, v in enumerate(effectifs):
    axes[0].text(i, v + 0.1, str(v), ha="center", fontsize=11, fontweight="bold")

# --- Graphe 2 : Diagramme en barres des fréquences (%) ---
axes[1].bar(labels, freq_pct, color=["#9FE1CB", "#1D9E75", "#085041"],
            edgecolor="white", width=0.6)
axes[1].set_title("Fréquences relatives (%)")
axes[1].set_xlabel("Heures de sommeil")
axes[1].set_ylabel("Fréquence (%)")
for i, v in enumerate(freq_pct):
    axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

# --- Graphe 3 : Fréquences cumulées ---
F_cum = np.cumsum(frequences) * 100
bornes = [4, 6, 8]   # bornes supérieures des classes
axes[2].plot([0] + bornes, [0] + list(F_cum), marker="o",
             color="#7F77DD", linewidth=2, markersize=7)
axes[2].axhline(y=50, color="#D85A30", linestyle="--", linewidth=1.2,
                label="50% → médiane")
axes[2].axvline(x=mediane, color="#D85A30", linestyle="--", linewidth=1.2)
axes[2].set_title("Fréquences cumulées (%)")
axes[2].set_xlabel("Heures de sommeil")
axes[2].set_ylabel("Fréquence cumulée (%)")
axes[2].set_xticks([0, 4, 6, 8])
axes[2].legend(fontsize=9)
for x, y in zip([0]+bornes, [0]+list(F_cum)):
    axes[2].text(x + 0.1, y + 1.5, f"{y:.1f}%", fontsize=9)

plt.tight_layout()
plt.savefig("analyse_sommeil.png", dpi=150, bbox_inches="tight")
plt.show()