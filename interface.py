import streamlit as st
from pulp import *
import pandas as pd

#  CONFIGURATION DE L'INTERFACE
st.set_page_config(page_title="Optimiseur AI Factory", layout="wide")

# Injection de CSS pour une interface sobre et professionnelle (Bleu nuit et Blanc)
st.markdown("""
    <style>
    /* Fond sombre professionnel */
    .stApp {
        background-color: #0c121c;
        color: #ffffff;
    }

    /* Titres en bleu acier */
    h1, h2, h3 {
        color: #7ab8ff !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 300;
        letter-spacing: 1px;
    }

    /* Tableaux et dataframes */
    .stDataFrame {
        background-color: #161e2b;
        border-radius: 5px;
    }

    /* Barre latérale */
    section[data-testid="stSidebar"] {
        background-color: #111926 !important;
        border-right: 1px solid #2d3a4f;
    }

    /* Métriques */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'monospace';
    }

    [data-testid="stMetric"] {
        background-color: #161e2b;
        border-left: 3px solid #7ab8ff;
        padding: 10px;
    }

    /* Barres de progression */
    .stProgress > div > div > div > div {
        background-color: #7ab8ff;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Système d'Optimisation de Production AI Factory")
st.markdown("Outil d'aide à la décision pour l'allocation des ressources de calcul.")
st.markdown("---")

# SECTION 1 : PARAMÈTRES (BARRE LATÉRALE)
with st.sidebar:
    st.header("Paramètres du Système")
    # Paramètres de capacité des ressources (GPU)
    st.subheader("Capacité des GPU (Heures)")
    dispo_a100 = st.slider("Capacité A100", 50, 400, 200)
    dispo_v100 = st.slider("Capacité V100", 50, 500, 300)
    dispo_t4 = st.slider("Capacité T4", 100, 1000, 500)
    # Paramètres de la demande client
    st.subheader("Demande Clients (Unités)")
    cmd_bert = st.number_input("Modèles BERT", 0, 100, 30)
    cmd_resnet = st.number_input("Modèles ResNet", 0, 100, 50)
    cmd_lstm = st.number_input("Modèles LSTM", 0, 100, 40)


# SECTION 2 : MOTEUR D'OPTIMISATION (PuLP)
def optimiser_production():
    # Définition des ensembles (Modèles et GPU)
    models = ["BERT", "ResNet", "LSTM"]
    gpus = ["A100", "V100", "T4"]

    # Dictionnaire des coûts unitaires (Modèle, GPU)
    cost = {
        ("BERT", "A100"): 20, ("BERT", "V100"): 35, ("BERT", "T4"): 80,
        ("ResNet", "A100"): 15, ("ResNet", "V100"): 20, ("ResNet", "T4"): 40,
        ("LSTM", "A100"): 5, ("LSTM", "V100"): 8, ("LSTM", "T4"): 12,
    }
    # Dictionnaire des temps d'exécution par modèle sur chaque GPU
    time_per_model = {
        ("BERT", "A100"): 4.0, ("BERT", "V100"): 8.75, ("BERT", "T4"): 40.0,
        ("ResNet", "A100"): 3.0, ("ResNet", "V100"): 5.0, ("ResNet", "T4"): 20.0,
        ("LSTM", "A100"): 1.0, ("LSTM", "V100"): 2.0, ("LSTM", "T4"): 6.0,
    }

    # Initialisation du problème de minimisation
    prob = LpProblem("Optimisation_Production", LpMinimize)
    # Définition des variables de décision (Entières car on ne divise pas un entraînement)
    x = LpVariable.dicts("x", [(m, g) for m in models for g in gpus], lowBound=0, cat='Integer')
    # Définition de la fonction objective : Minimiser le coût total
    prob += lpSum(cost[m, g] * x[m, g] for m in models for g in gpus)
    # Contraintes de Demande : Chaque modèle doit être produit selon la commande
    for m in models:
        prob += lpSum(x[m, g] for g in gpus) == {"BERT": cmd_bert, "ResNet": cmd_resnet, "LSTM": cmd_lstm}[m]
    # Contraintes de Capacité : Le temps utilisé sur chaque GPU ne doit pas dépasser le disponible
    disponibilites = {"A100": dispo_a100, "V100": dispo_v100, "T4": dispo_t4}
    for g in gpus:
        prob += lpSum(time_per_model[m, g] * x[m, g] for m in models) <= disponibilites[g]

    # Contraintes Métier Spécifiques (Quotas et exigences techniques)
    prob += x["BERT", "A100"] + x["BERT", "V100"] >= 10 # Minimum sur GPU haute performance
    prob += x["ResNet", "T4"] <= 20                     # Limitation technique sur T4
    prob += x["LSTM", "V100"] + x["LSTM", "T4"] >= 15   # Distribution minimale
    prob += x["BERT", "A100"] + x["ResNet", "A100"] + x["LSTM", "A100"] <= 25  # Quota global sur A100
    # Lancement du solveur
    prob.solve()
    return prob, x, models, gpus, time_per_model, disponibilites


# SECTION 3 : AFFICHAGE DES RÉSULTATS
status, x, models, gpus, time_per_model, disponibilites = optimiser_production()

if LpStatus[status.status] == 'Optimal':
    # Affichage du KPI principal (Coût Total)
    st.metric("Coût Total d'Entraînement", f"{int(value(status.objective))} €")
    # Visualisation de la charge des ressources
    st.markdown("### Répartition des Charges GPU")
    cols = st.columns(3)
    for i, g in enumerate(gpus):
        heures_utilisees = sum(value(x[m, g]) * time_per_model[m, g] for m in models)
        taux = heures_utilisees / disponibilites[g]
        with cols[i]:
            st.write(f"Nœud {g}")
            st.progress(min(taux, 1.0))
            st.caption(f"{heures_utilisees:.1f}h utilisées sur {disponibilites[g]}h")
    # Présentation des résultats sous forme de tableau
    st.markdown("### Matrice d'Affectation des Modèles")
    donnees_tableau = []
    for m in models:
        ligne = {"Type de Modèle": m}
        for g in gpus:
            ligne[g] = int(value(x[m, g]))
        donnees_tableau.append(ligne)

    st.table(pd.DataFrame(donnees_tableau))
# Message d'alerte si aucune solution n'est mathématiquement possible
else:
    st.error(
        "Aucune solution optimale n'a été trouvée avec les paramètres actuels. Veuillez augmenter les capacités ou réduire la demande.")