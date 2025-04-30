import streamlit as st
import joblib

# Load model, vectorizer, and label encoder
model = joblib.load("mbti_model_xgboost.pkl")
vectorizer = joblib.load("mbti_vectorizer.pkl")
label_encoder = joblib.load("mbti_label_encoder.pkl")

# Streamlit App UI
st.title("🧠 MBTI Personality Predictor")
st.write("In a few sentences, describe who you are—your background, passions, skills, or anything that makes you unique. This helps us understand you better!")
# st.write("How do you typically recharge after a long day—spending time alone or socializing with others?")
# st.write("Do you prefer making detailed plans before starting a project or diving right in and figuring things out as you go?")
# st.write("When faced with a decision, do you rely more on facts and logic or on emotions and personal values?")
# st.write("Do you feel more comfortable focusing on present realities or imagining future possibilities?")
# st.write("How do you approach conflict—avoiding it or addressing it directly?")
# st.write("Do you find yourself drawn to tasks that require creativity and innovation, or ones with clear procedures and goals?")
# st.write("When communicating, do you prefer expressing thoughts clearly and concisely, or sharing abstract ideas and feelings?")
# st.write("In a group setting, do you tend to take charge and organize or go with the flow and adapt?")

user_input = st.text_area("Your Description", height=200)

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
