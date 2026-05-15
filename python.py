import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#__Données__ intervalles

heures = [5]*8 + [7]*9  # 8 personnes entre 4h et 6h, 9 personnes entre 6h et 8h

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

# représentation graphique
plt.hist(heures, bins=[4, 6, 8], edgecolor='black')
plt.title("Distribution des heures de sommeil")
plt.xlabel("Heures de sommeil")
plt.ylabel("Nombre de personnes")
plt.xticks([4, 5, 6, 7, 8])
plt.show()

#boite à moustaches
plt.boxplot(heures, vert=False)
plt.title("Boîte à moustaches des heures de sommeil")
plt.xlabel("Heures de sommeil")
plt.yticks([1], [''])
plt.show()

#courbe de densité
x = np.linspace(4, 8, 100)
kde = stats.gaussian_kde(heures)
plt.plot(x, kde(x), label='Densité de probabilité')
plt.title("Courbe de densité des heures de sommeil")
plt.xlabel("Heures de sommeil")
plt.ylabel("Densité")
plt.legend()
plt.show()



