from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value

# ============================
# 1. Données du problème
# ============================
models = ["BERT", "ResNet", "LSTM"]  # Liste des modèles à entraîner
gpus = ["A100", "V100", "T4"]        # Liste des GPU disponibles

# Coût d'entraînement d'un modèle sur chaque GPU (en euros)
cost = {
    ("BERT", "A100"): 20, ("BERT", "V100"): 35, ("BERT", "T4"): 80,
    ("ResNet", "A100"): 15, ("ResNet", "V100"): 20, ("ResNet", "T4"): 40,
    ("LSTM", "A100"): 5, ("LSTM", "V100"): 8, ("LSTM", "T4"): 12,
}

# Temps nécessaire pour entraîner un modèle sur chaque GPU (en heures)
time_per_model = {
    ("BERT", "A100"): 4, ("BERT", "V100"): 8.75, ("BERT", "T4"): 40,
    ("ResNet", "A100"): 3, ("ResNet", "V100"): 5, ("ResNet", "T4"): 20,
    ("LSTM", "A100"): 1, ("LSTM", "V100"): 2, ("LSTM", "T4"): 6,
}

# Demande minimale pour chaque modèle
demand = {"BERT": 30, "ResNet": 50, "LSTM": 40}

# Disponibilité horaire de chaque GPU
hours_available = {"A100": 200, "V100": 300, "T4": 500}

# ============================
# 2. Définition du problème
# ============================
prob = LpProblem("Minimisation_Cout_Entrainement_IA", LpMinimize)  # Problème de minimisation

# ============================
# 3. Variables de décision
# ============================
# x[(m,g)] = nombre de modèles m à entraîner sur le GPU g
# lowBound=0 : pas de valeurs négatives
# cat='Integer' : variables entières car on ne peut pas entraîner une fraction de modèle
x = LpVariable.dicts("x", [(m, g) for m in models for g in gpus], lowBound=0, cat='Integer')

# ============================
# 4. Fonction objectif
# ============================
# Minimiser le coût total d'entraînement
prob += lpSum(cost[m, g] * x[m, g] for m in models for g in gpus)

# ============================
# 5. Contraintes
# ============================

# 5.1 Contraintes de demande : chaque modèle doit être entraîné au moins autant que la demande
for m in models:
    prob += lpSum(x[m, g] for g in gpus) == demand[m]

# 5.2 Contraintes de temps GPU : ne pas dépasser le nombre d'heures disponibles
for g in gpus:
    prob += lpSum(time_per_model[m, g] * x[m, g] for m in models) <= hours_available[g]

# 5.3 Contraintes de quotas spécifiques
prob += x["BERT", "A100"] + x["BERT", "V100"] >= 10                    # Minimum BERT sur A100+V100
prob += x["ResNet", "T4"] <= 20                                         # Maximum ResNet sur T4
prob += x["LSTM", "V100"] + x["LSTM", "T4"] >= 15                        # Minimum LSTM sur V100+T4
prob += x["BERT", "A100"] + x["ResNet", "A100"] + x["LSTM", "A100"] <= 25  # Quota global A100

# ============================
# 6. Résolution du problème
# ============================
prob.solve()  # Appel du solveur (CBC par défaut)

# ============================
# 7. Affichage du tableau simple d'allocation
# ============================
print("="*45)
print("ALLOCATION DES MODÈLES")
print("="*45)
print(f"{'Modèle/GPU':<12} {'A100':<8} {'V100':<8} {'T4':<8}")
print("-"*45)

for m in models:
    ligne = f"{m:<12}"
    for g in gpus:
        valeur = int(value(x[m, g]))  # Conversion en entier pour affichage
        ligne += f" {valeur:<8}"
    print(ligne)

print("="*45)
print(f"Coût total optimal: {int(value(prob.objective))} €")

# ============================
# 8. Analyse post-optimisation
# ============================
print("\n" + "="*55)
print("ANALYSE POST-OPTIMISATION")
print("="*55)

# 8.1 Modèles entièrement alloués
print("\n--- Allocation optimale par modèle ---")
for m in models:
    total_alloc = sum(value(x[m, g]) for g in gpus)
    print(f"{m}: {int(total_alloc)} unités allouées (demande = {demand[m]})")

# 8.2 GPU saturés ou non
print("\n--- GPU saturés ---")
for g in gpus:
    usage = sum(time_per_model[m, g] * value(x[m, g]) for m in models)
    print(f"{g}: {usage}/{hours_available[g]} heures utilisées", end='')
    if abs(usage - hours_available[g]) < 1e-5:
        print(" → GPU saturé")
    else:
        print(" → GPU non saturé")

# 8.3 Vérification des quotas
print("\n--- Contraintes de quotas ---")
if value(x["BERT","A100"] + x["BERT","V100"]) >= 10:
    print("Quota BERT sur A100+V100 respecté")
if value(x["ResNet","T4"]) <= 20:
    print("Quota ResNet sur T4 respecté")
if value(x["LSTM","V100"] + x["LSTM","T4"]) >= 15:
    print("Quota LSTM sur V100+T4 respecté")
if value(x["BERT","A100"] + x["ResNet","A100"] + x["LSTM","A100"]) <= 25:
    print("Quota global A100 respecté")

# 8.4 Interprétation des coûts réduits et des prix du dual
print("\n--- Interprétation ---")
print("1. Les coûts réduits indiquent l’impact sur le coût total si une variable non utilisée était augmentée de 1 unité.")
print("2. Les prix du dual montrent combien le coût total augmenterait si la disponibilité d’un GPU augmentait d’une unité.")
print("\n→ Ces informations permettent de décider si investir dans plus de GPU ou ajuster les quotas est rentable.")

print("\nPost-optimisation terminée. ")


# ================================
# Analyse du goulot d'étranglement
# ================================

# Affichage d'un titre pour rendre la sortie plus lisible
print("\n" + "="*45)
print("ANALYSE DES GOULOTS D'ÉTRANGLEMENT")
print("="*45)

# Dictionnaire pour stocker le nombre total d'heures utilisées par chaque GPU
used_hours = {}

# Dictionnaire pour stocker le taux d'utilisation (en pourcentage) de chaque GPU
utilization = {}

# Parcours de chaque type de GPU
for g in gpus:
    
    # Calcul du nombre total d'heures utilisées par le GPU g
    # Pour chaque modèle m, on multiplie :
    # - x[m, g] : le nombre de tâches du modèle m affectées au GPU g
    # - time_per_model[m, g] : le temps nécessaire pour exécuter le modèle m sur le GPU g
    used_hours[g] = sum(
        value(x[m, g]) * time_per_model[m, g]
        for m in models
    )

    # Calcul du taux d'utilisation du GPU g en pourcentage
    # (heures utilisées / heures disponibles) * 100
    utilization[g] = used_hours[g] / hours_available[g] * 100

    # Affichage des résultats pour le GPU g
    print(
        f"{g} : {used_hours[g]:.2f} h utilisées / "
        f"{hours_available[g]} h "
        f"({utilization[g]:.2f} %)"
    )

# Ligne de séparation pour la lisibilité
print("="*45)

# Identification du GPU le plus sollicité
# c'est-à-dire celui ayant le taux d'utilisation maximal
bottleneck = max(utilization, key=utilization.get)

# Affichage du GPU goulot d'étranglement
print(f" GPU Goulot d'étranglement : {bottleneck}")