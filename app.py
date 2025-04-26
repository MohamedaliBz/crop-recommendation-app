import streamlit as st
import joblib
import numpy as np
import time

# Charger les fichiers nécessaires
model = joblib.load("crop_recommendation_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Configuration de la page
st.set_page_config(page_title="Recommandation de Culture", page_icon="🌱")
st.title("🌾 Système de recommandation de culture")
st.markdown("Entrez les paramètres du sol ci-dessous pour obtenir une recommandation de culture adaptée.")

# Champs d'entrée utilisateur
N = st.number_input("Azote (N)", min_value=0.0, step=1.0)
P = st.number_input("Phosphore (P)", min_value=0.0, step=1.0)
K = st.number_input("Potassium (K)", min_value=0.0, step=1.0)
temperature = st.number_input("Température (°C)", step=0.1)
humidity = st.number_input("Humidité (%)", min_value=0.0, max_value=100.0, step=0.1)
ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Précipitations (mm)", min_value=0.0, step=0.1)

# Bouton de prédiction
if st.button("Prédire la culture recommandée"):
    # Vérification des champs : tous doivent être strictement supérieurs à 0
    if any(val <= 0 for val in [N, P, K, temperature, humidity, ph, rainfall]):
        st.error("❌ Veuillez saisir des valeurs valides supérieures à zéro pour tous les champs.")
    else:
        try:
            # Transformation log1p sur K et rainfall
            K_log = np.log1p(K)
            rainfall_log = np.log1p(rainfall)

            # Standardisation des données utilisateur
            user_input = np.array([[N, P, K_log, temperature, humidity, ph, rainfall_log]])
            user_input_scaled = scaler.transform(user_input)

            # Ajout temporaire pour vérifier
            st.write("🔍 Données après log1p :", user_input)
            st.write("🔍 Données après standardisation :", user_input_scaled)

            # Indicateur de chargement
            with st.spinner("⏳ Prédiction en cours..."):
                time.sleep(2.5) # Simuler un délai de traitement

                prediction_encoded = model.predict(user_input_scaled)
                prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

                # Affichage du résultat
                st.success(f"✅ Culture recommandée : **{prediction_label}**")

                # Suggestions personnalisées
                suggestions = {
                    "apple": "🍎 Les pommiers poussent bien dans des climats tempérés avec des hivers froids.",
                    "banana": "🍌 Les bananiers ont besoin de chaleur et d'humidité.",
                    "blackgram": "🌱 Le black gram préfère les sols bien drainés et les climats chauds.",
                    "chickpea": "🥙 Les pois chiches poussent bien dans des sols légers et des climats secs.",
                    "coconut": "🥥 Les cocotiers nécessitent des climats tropicaux et des sols sableux.",
                    "coffee": "☕ Les caféiers poussent bien dans des climats tropicaux avec des sols riches et bien drainés.",
                    "cotton": "🧵 Le coton nécessite des sols bien drainés et des climats chauds.",
                    "grapes": "🍇 Le raisin préfère un sol sablonneux et du soleil.",
                    "jute": "🧵 Le jute pousse bien dans des sols alluviaux et des climats humides.",
                    "kidneybeans": "🥘 Les haricots rouges poussent bien dans des sols riches et bien drainés.",
                    "lentil": "🍲 Les lentilles préfèrent les sols légers et bien drainés.",
                    "maize": "🌽 Le maïs pousse bien dans des sols riches et bien drainés.",
                    "mango": "🥭 Les manguiers aiment les climats tropicaux secs.",
                    "mothbeans": "🌾 Les moth beans poussent bien dans des sols secs et des climats chauds.",
                    "mungbean": "🌱 Les mung beans préfèrent les sols bien drainés et les climats chauds.",
                    "muskmelon": "🍈 Les melons musqués poussent bien dans des sols sablonneux et des climats chauds.",
                    "orange": "🍊 Les orangers préfèrent les climats chauds et les sols bien drainés.",
                    "papaya": "🍈 Les papayers aiment les sols bien drainés et les climats chauds.",
                    "pigeonpeas": "🌾 Les pois d'Angole poussent bien dans des sols légers et des climats chauds.",
                    "pomegranate": "🍎 Les grenadiers préfèrent les sols bien drainés et les climats secs.",
                    "rice": "💧 Le riz aime les zones très humides avec beaucoup d'eau.",
                    "watermelon": "🍉 Les pastèques poussent bien dans des sols sablonneux et des climats chauds.",
                }

                st.info(f"ℹ️ Astuce : {suggestions.get(prediction_label.lower())}")

        except Exception as e:
            st.error(f"❌ Une erreur est survenue : {e}")
