import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import streamlit as st

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "credit_risk_dataset.csv")

df = pd.read_csv(file_path, sep=';')

# 1. Chargement du dataset
#df = pd.read_csv('credit_risk_dataset.csv', sep=';')

# 2. Exploration des données
print("Aperçu des données :")
print(df.head())
print("\nRésumé statistique :")
print(df.describe())
print("\nValeurs manquantes :")
print(df.isnull().sum())

# 3. Traitement des données
# Remplir les valeurs manquantes
# Remplacement des valeurs manquantes pour les colonnes numériques
num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Remplacement des valeurs manquantes pour les colonnes catégoriques par la valeur la plus fréquente
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Encodage des variables catégoriques
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col + '_encoded'] = le.fit_transform(df[col])


# Drop the original categorical columns
df = df.drop(columns=cat_cols)

# Rename encoded columns to the original names for consistency
for col in cat_cols:
    df = df.rename(columns={col + '_encoded': col})


# Séparation des variables indépendantes et cible
X = df.drop(columns=['loan_status'])  # Supposons que la colonne cible est "risk"
y = df['loan_status']

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Entraînement des modèles
# Modèle de Régression Logistique
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Modèle Arbre de Décision
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# 5. Évaluation des modèles
def evaluate_model(model_name, y_true, y_pred):
    print(f"\nÉvaluation du modèle : {model_name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, y_pred))
    print("Rapport de classification:\n", classification_report(y_true, y_pred))
    print("Matrice de confusion:\n", confusion_matrix(y_true, y_pred))

evaluate_model("Régression Logistique", y_test, y_pred_log)
evaluate_model("Arbre de Décision", y_test, y_pred_dt)

# Sauvegarde du meilleur modèle
joblib.dump(dt_model, 'best_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# 6. Interface Streamlit
st.title("Prédiction du Risque de Crédit")

# Charger le modèle
model = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')

# Interface utilisateur
def predict_risk(features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return "Client Risqué" if prediction[0] == 1 else "Client Non Risqué"

# Formulaire utilisateur
input_data = []
for col in X.columns:
    value = st.number_input(f"{col}", value=float(df[col].median()))
    input_data.append(value)

if st.button("Prédire"):
    result = predict_risk(input_data)
    st.write(f"Résultat : {result}")
