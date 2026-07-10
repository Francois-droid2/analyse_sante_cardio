import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Fichier de stockage des données sur le serveur
DATA_FILE = "donnees_sante_cardiaque.csv"

def charger_donnees():
    if os.path.exists(DATA_FILE):
        try:
            return pd.read_csv(DATA_FILE)
        except:
            pass
    # Échantillon initial si la base est vide au premier lancement
    np.random.seed(42)
    ages = np.random.randint(25, 75, size=50)
    genres = np.random.choice(["Masculin", "Féminin"], size=50)
    tensions = np.random.randint(110, 160, size=50)
    cholesterols = np.round(np.random.uniform(1.6, 3.2, size=50), 2)
    fumeurs = np.random.choice(["Oui", "Non"], size=50)
    return pd.DataFrame({
        "Age": ages, "Genre": genres, "Tension": tensions, 
        "Cholesterol": cholesterols, "Fumeur": fumeurs
    })

# Charger la base de données globale
df = charger_donnees()

# --- CONFIGURATION DU MENU DANS LA BARRE LATÉRALE ---
st.sidebar.title("Menu de Navigation")
choix_menu = st.sidebar.radio(
    "Accéder aux sections :",
    options=[
        "1. Formulaire de Collecte",
        "2. Analyse Descriptive",
        "3. Apprentissage Supervisé",
        "4. Apprentissage Non Supervisé",
        "5. Historique des Données"
    ]
)

# En-tête textuel fixe
st.title("Application d'Analyse de la Santé Cardiaque")
st.write("Collecte et modélisation statistique — Conforme aux exigences académiques.")
st.write("---")

# ==========================================
# SECTIONS DU MENU
# ==========================================

if choix_menu == "1. Formulaire de Collecte":
    st.header("Formulaire de Collecte des Données")
    st.write("Entrez les paramètres physiologiques pour alimenter la base de données :")
    
    with st.form(key="form_sante", clear_on_submit=True):
        age = st.number_input("Âge (ans)", min_value=18, max_value=100, value=45, step=1)
        genre = st.radio("Genre", options=["Masculin", "Féminin"])
        tension = st.number_input("Pression Artérielle Systolique (mmHg)", min_value=80, max_value=220, value=130, step=1)
        cholesterol = st.number_input("Taux de Cholestérol Total (g/L)", min_value=1.0, max_value=5.0, value=2.2, step=0.1)
        fumeur = st.checkbox("Fumeur actif")
        soumettre = st.form_submit_button(label="Enregistrer les données")

    if soumettre:
        nouvelle_entree = {
            "Age": int(age), 
            "Genre": str(genre), 
            "Tension": int(tension), 
            "Cholesterol": float(cholesterol), 
            "Fumeur": "Oui" if fumeur else "Non"
        }
        df = pd.concat([df, pd.DataFrame([nouvelle_entree])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        st.success("Données enregistrées ! Vous pouvez consulter les analyses dans les autres sections du menu.")

elif choix_menu == "2. Analyse Descriptive":
    st.header("Analyse Descriptive Globale")
    total = len(df)
    st.subheader(f"Effectif total de la population d'étude : {total} personnes")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Indicateurs de tendance centrale (Moyennes) :**")
        st.write(f"- **Âge moyen :** {df['Age'].mean():.1f} ans")
        st.write(f"- **Tension moyenne :** {df['Tension'].mean():.1f} mmHg")
        st.write(f"- **Cholestérol moyen :** {df['Cholesterol'].mean():.2f} g/L")
    with col2:
        st.markdown("**Proportions des variables qualitatives :**")
        nb_hommes = len(df[df["Genre"] == "Masculin"])
        nb_fumeurs = len(df[df["Fumeur"] == "Oui"])
        st.write(f"- **Hommes :** {(nb_hommes / total) * 100:.1f}% ({nb_hommes} individus)")
        st.write(f"- **Femmes :** {((total - nb_hommes) / total) * 100:.1f}% ({total - nb_hommes} individus)")
        st.write(f"- **Fumeurs :** {(nb_fumeurs / total) * 100:.1f}% ({nb_fumeurs} individus)")

elif choix_menu == "3. Apprentissage Supervisé":
    st.header("Apprentissage Supervisé : Classification du Risque")
    st.write("**Algorithme de calcul :** Régression Logistique")
    
    df_ml = df.copy()
    df_ml["Genre_Num"] = df_ml["Genre"].apply(lambda x: 1 if x == "Masculin" else 0)
    df_ml["Fumeur_Num"] = df_ml["Fumeur"].apply(lambda x: 1 if x == "Oui" else 0)
    X = df_ml[["Age", "Tension", "Cholesterol", "Genre_Num", "Fumeur_Num"]]
    
    y = ((df_ml["Tension"] > 140) & (df_ml["Cholesterol"] > 2.4) | (df_ml["Age"] > 60) & (df_ml["Fumeur_Num"] == 1)).astype(int)
    
    if len(np.unique(y)) > 1:
        modele_sup = LogisticRegression()
        modele_sup.fit(X, y)
        
        st.markdown("### Simulation de diagnostic")
        t_age = st.slider("Choisir l'âge", 18, 100, 45)
        t_genre = st.radio("Choisir le genre", ["Masculin", "Féminin"])
        t_tension = st.slider("Choisir la tension (mmHg)", 80, 220, 120)
        t_chol = st.slider("Choisir le taux de cholestérol (g/L)", 1.0, 5.0, 2.0, step=0.1)
        t_fumeur = st.checkbox("Cocher si fumeur")
        
        profil = np.array([[t_age, t_tension, t_chol, 1 if t_genre == "Masculin" else 0, 1 if t_fumeur else 0]])
        
        prediction = modele_sup.predict(profil)[0]
        probabilite = modele_sup.predict_proba(profil)[0][1] * 100
        
        if prediction == 1:
            st.error(f"⚠️ Classe prédite : RISQUE CARDIAQUE ÉLEVÉ (Probabilité calculée : {probabilite:.1f}%)")
        else:
            st.success(f"✅ Classe prédite : RISQUE CARDIAQUE FAIBLE (Probabilité calculée : {probabilite:.1f}%)")
    else:
        st.info("Ajoutez des profils plus variés dans le formulaire pour démarrer l'apprentissage.")

elif choix_menu == "4. Apprentissage Non Supervisé":
    st.header("Apprentissage Non Supervisé : Clustering")
    st.write("**Algorithme de calcul :** K-Means (Partitionnement en 2 groupes distincts)")
    
    df_ml = df.copy()
    df_ml["Genre_Num"] = df_ml["Genre"].apply(lambda x: 1 if x == "Masculin" else 0)
    df_ml["Fumeur_Num"] = df_ml["Fumeur"].apply(lambda x: 1 if x == "Oui" else 0)
    X = df_ml[["Age", "Tension", "Cholesterol", "Genre_Num", "Fumeur_Num"]]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df_ml["Cluster"] = kmeans.fit_predict(X_scaled)
    
    st.subheader("Segmentation automatique de l'échantillon :")
    for i in range(2):
        cluster_data = df_ml[df_ml["Cluster"] == i]
        st.write(f"**Groupe d'individus n°{i}** ({len(cluster_data)} personnes) : "
                 f"Âge moyen = {cluster_data['Age'].mean():.1f} ans, "
                 f"Tension moyenne = {cluster_data['Tension'].mean():.1f} mmHg, "
                 f"Fumeurs = {len(cluster_data[cluster_data['Fumeur'] == 'Oui'])} personnes.")

elif choix_menu == "5. Historique des Données":
    st.header("Registre des Données Anonymisées")
    st.write("Voici la base de données brute exploitée en temps réel par les algorithmes :")
    st.dataframe(df)
