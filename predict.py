import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the TF-IDF vectorizer
vectorizer_path = "vectorizer.pkl"
vect = joblib.load(vectorizer_path)

def predict_text(text, models):
    text = vect.transform([text])
    predictions = {}
    for model_name, model_path in models.items():
        model = joblib.load(model_path)
        prediction = model.predict(text)
        if prediction[0] == "Hate Speech" or prediction[0] == "Offensive Language":
            predictions[model_name] = "Offensive"
        else:
            predictions[model_name] = "Not Offensive"
    return predictions
