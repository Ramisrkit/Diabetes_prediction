from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Read input values from form
    inputs = [float(x) for x in request.form.values()]
    
    # Convert to 2D array
    final_features = np.array([inputs])
    
    # Apply scaling
    final_features = scaler.transform(final_features)
    
    # Make prediction
    prediction = model.predict(final_features)[0]
    
    # Convert numeric result to text
    output = "Diabetes" if prediction == 1 else "No Diabetes"

    return render_template("index.html", prediction_text=f"Prediction: {output}")

if __name__ == "__main__":
    app.run(debug=True)
