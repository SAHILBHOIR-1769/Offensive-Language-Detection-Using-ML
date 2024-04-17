from flask import Flask, render_template, request
from predict import predict_text

app = Flask(__name__)

models = {
    "Logistic Regression": "models/trained_logreg_model.pkl",
    "Decision Tree": "models/trained_dtree_model.pkl",
    "SVM": "models/trained_svm_model.pkl",
    "Hyper-tuned Logistic Regression": "models/best_logreg_model.pkl",
    "Ensemble": "models/trained_ensemble_model.pkl"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    predictions = predict_text(text, models)
    return render_template('index.html', text=text, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
