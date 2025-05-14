import streamlit as st
import joblib

# Load model, vectorizer, and label encoder
model = joblib.load("mbti_model_xgboost.pkl")
vectorizer = joblib.load("mbti_vectorizer.pkl")
label_encoder = joblib.load("mbti_label_encoder.pkl")

# Streamlit App UI
st.title("🧠 PsyQ : MBTI Personality Predictor")
st.write("In a few sentences, describe who you are—your background, passions, skills, or anything that makes you unique. This helps us understand you better!")

user_input = st.text_area("Your Description", height=100)

if st.button("Predict MBTI Type"):
    if user_input.strip():
        # Vectorize input
        input_vector = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(input_vector)[0]
        prediction_label = label_encoder.inverse_transform([prediction])[0]
        
        # Confidence (probabilities)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_vector)[0]
            confidence = max(proba) * 100
            st.success(f"🧬 Predicted MBTI Type: **{prediction_label}** ({confidence:.2f}% confidence)")
        else:
            st.success(f"🧬 Predicted MBTI Type: **{prediction_label}**")
    else:
        st.warning("Please enter some text to predict.")
st.write("Know your type in a single swipe of text.")
