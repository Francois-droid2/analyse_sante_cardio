import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Fichier local de stockage des données
DATA_FILE = "donnees_sante_cardiaque.csv"

def charger_donnees():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        # Échantillon initial simulé pour éviter les bugs au premier lancement
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

# Charger la base globale
df = charger_donnees()

# --- BARRE LATÉRALE : MENU DE NAVIGATION (TEXTUEL) ---
st.sidebar.title("Navigation")
choix_menu = st.sidebar.radio(
    "Sélectionnez une section :",
    options=[
        "1. Collecte des données",
        "2. Analyse Descriptive",
        "3. Apprentissage Supervisé",
        "4. Apprentissage Non Supervisé",
        "5. Base de données complète"
    ]
)

# En-tête permanent
st.title("Plateforme d'Analyse Avancée de la Santé Cardiaque")
st.write("Projet académique — Collecte & Machine Learning.")
st.write("---")


# ==========================================
# RECOUPEMENT DES PAGES SELON LE MENU
# ==========================================

# --- PAGE 1 : COLLECTE ---
if choix_menu == "1. Collecte des données":
    st.header("Formulaire de Collecte des Données")
    st.write("Veuillez renseigner les paramètres physiologiques ci-dessous :")
    
    with st.form(key="form_sante", clear_on_submit=False):
        age = st.number_input("Âge", min_value=18, max_value=100, value=45)
        genre = st.radio("Genre", options=["Masculin", "Féminin"])
        tension = st.number_input("Tension Artérielle Systolique (mmHg)", min_value=80, max_value=220, value=130)
        cholesterol = st.number_input("Taux de Cholestérol Total (g/L)", min_value=1.0, max_value=5.0, value=2.2, step=0.1)
        fumeur = st.checkbox("Fumeur actif")
        
        soumettre = st.form_submit_button(label="Enregistrer et Mettre à jour l'application")

    if soumettre:
        nouvelle_entree = {
            "Age": age, "Genre": genre, "Tension": tension, 
            "Cholesterol": cholesterol, "Fumeur": "Oui" if fumeur else "Non"
        }
        df = pd.concat([df, pd.DataFrame([nouvelle_entree])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        st.success("Données enregistrées avec succès ! Allez dans les autres menus pour voir l'analyse.")


# --- PAGE 2 : ANALYSE DESCRIPTIVE ---
elif choix_menu == "2. Analyse Descriptive":
    st.header("Analyse Descriptive de la Population")
    total = len(df)
    st.subheader(f"Taille actuelle de l'échantillon : {total} profils")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Statistiques des variables numériques (Moyennes) :**")
        st.write(f"- Moyenne d'âge : {df['Age'].mean():.1f} ans")
        st.write(f"- Tension moyenne : {df['Tension'].mean():.1f} mmHg")
        st.write(f"- Cholestérol moyen : {df['Cholesterol'].mean():.2f} g/L")
    with col2:
        st.markdown("**Répartition des variables catégorielles (Proportions) :**")
        pct_hommes = (len(df[df["Genre"] == "Masculin"]) / total) * 100
        pct_fumeurs = (len(df[df["Fumeur"] == "Oui"]) / total) * 100
        st.write(f"- Proportion d'Hommes : {pct_hommes:.1f}%")
        st.write(f"- Proportion de Fumeurs : {pct_fumeurs:.1f}%")


# --- PRÉPARATION DU DATASET POUR LE ML ---
# Encodage requis pour les calculs des sections 3 et 4
df_ml = df.copy()
df_ml["Genre_Num"] = df_ml["Genre"].apply(lambda x: 1 if x == "Masculin" else 0)
df_ml["Fumeur_Num"] = df_ml["Fumeur"].apply(lambda x: 1 if x == "Oui" else 0)
X = df_ml[["Age", "Tension", "Cholesterol", "Genre_Num", "Fumeur_Num"]]


# --- PAGE 3 : SUPERVISÉ ---
elif choix_menu == "3. Apprentissage Supervisé":
    st.header("Apprentissage Supervisé : Modèle de Classification")
    st.write("**Algorithme utilisé :** Régression Logistique")
    
    # Définition médicale simulée pour la cible Y (Risque)
    y = ((df_ml["Tension"] > 140) & (df_ml["Cholesterol"] > 2.4) | (df_ml["Age"] > 60) & (df_ml["Fumeur_Num"] == 1)).astype(int)
    
    if len(np.unique(y)) > 1:
        modele_sup = LogisticRegression()
        modele_sup.fit(X, y)
        
        st.markdown("### Tester le modèle sur un profil personnalisé")
        test_age = st.slider("Âge pour le test", 18, 100, 50)
        test_genre = st.radio("Genre pour le test", ["Masculin", "Féminin"])
        test_tension = st.slider("Tension (mmHg)", 80, 220, 130)
        test_chol = st.slider("Cholestérol (g/L)", 1.0, 5.0, 2.2, step=0.1)
        test_fumeur = st.checkbox("Fumeur")
        
        profil_test = np.array([[test_age, test_tension, test_chol, 1 if test_genre == "Masculin" else 0, 1 if test_fumeur else 0]])
        
        prediction = modele_sup.predict(profil_test)[0]
        probabilite = modele_sup.predict_proba(profil_test)[0][1] * 100
        
        if prediction == 1:
            st.error(f"⚠️ Diagnostic : Profil à RISQUE CARDIAQUE ÉLEVÉ (Probabilité : {probabilite:.1f}%)")
        else:
            st.success(f"✅ Diagnostic : Profil à RISQUE CARDIAQUE FAIBLE (Probabilité de risque : {probabilite:.1f}%)")
    else:
        st.info("Échantillon insuffisant pour entraîner le modèle. Ajoutez d'autres profils via le formulaire.")


# --- PAGE 4 : NON SUPERVISÉ ---
elif choix_menu == "4. Apprentissage Non Supervisé":
    st.header("Apprentissage Non Supervisé : Clustering K-Means")
    st.write("L'algorithme segmente automatiquement la population globale en 2 groupes distincts sans étiquettes préalables.")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df_ml["Cluster"] = kmeans.fit_predict(X_scaled)
    
    st.subheader("Caractérisation textuelle des groupes (Clusters) générés :")
    for i in range(2):
        cluster_data = df_ml[df_ml["Cluster"] == i]
        st.write(f"**Groupe {i}** ({len(cluster_data)} individus) : "
                 f"Âge moyen = {cluster_data['Age'].mean():.1f} ans, "
                 f"Tension moyenne = {cluster_data['Tension'].mean():.1f} mmHg, "
                 f"Fumeurs = {len(cluster_data[cluster_data['Fumeur'] == 'Oui'])} personnes.")


# --- PAGE 5 : DONNÉES BRUTES ---
elif choix_menu == "5. Base de données complète":
    st.header("Registre des données stockées")
    st.write("Voici l'intégralité des données anonymes collectées par le serveur.")
    st.dataframe(df)
