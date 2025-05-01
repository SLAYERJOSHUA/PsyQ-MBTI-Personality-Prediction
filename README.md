# 🧠 PsyQ: MBTI Personality Predictor

PsyQ is an intelligent personality prediction system that identifies an individual's MBTI type based on natural language input. Simply provide a paragraph describing yourself, and PsyQ will analyze your text to determine your personality type from the 16 MBTI types using advanced NLP and classification techniques. Trained on augmented keyword data, it uses machine learning models to deliver fast, accurate predictions through a simple Streamlit interface.

**🤖 STREAMLIT APP** : https://psyq-mbti-personality-prediction-ukncrba52cmetgjfmyds8g.streamlit.app/


![image](https://github.com/user-attachments/assets/f960a0a4-1741-4e8a-858c-63bebd6f4625)


## 🚀 Features

- 🔍 Predicts one of the 16 MBTI personality types from text input
- 📚 Keyword-augmented custom dataset for improved performance
- 🤖 Multiple machine learning models tested (XGBoost, SVM, Random Forest)
- 🧩 Supports n-gram analysis and preprocessing for better accuracy
- 🧠 Option to integrate transformer-based models like DistilBERT
- 💬 User-friendly input: just enter a paragraph about yourself!

## 📦 Requirements

- Python 3.7+
- pandas
- scikit-learn
- xgboost
- nltk
- joblib

##  🛠 How It Works
- **Input**: A user submits a short paragraph describing their behavior, thoughts, or values.
- **Preprocessing**: Text is cleaned, tokenized, and processed using n-grams and vectorization (TF-IDF).
- **Model**: Pre-trained models classify the text into one of the 16 MBTI types.
- **Output**: The predicted personality type is returned.

## 🧪 Example Usage
**Paragraph Input**: *I enjoy helping people and often find myself reflecting deeply on my emotions. I prefer meaningful conversations over small talk and care about                              harmony and authenticity.*

**Output**: ***INFP***

## 🧠 Model Accuracy
**📈 Current accuracy**: ~61% using classical ML models
*🚀 Option to upgrade to transformer-based models (DistilBERT) for better performance*
📊 Augmented dataset with ~2000 keywords/type for robustness. Data augmented using keyword synonyms and n-gram tokenization
 Accuracy

## MBTI types
![image](https://github.com/user-attachments/assets/8c2e67e4-44e7-45ea-a808-fabbec350066)

# SNIPPETS

![image](https://github.com/user-attachments/assets/ee7d3a59-7f08-421d-96e0-eda7d346edd7)





