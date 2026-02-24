from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open("house_model.pkl", "rb"))

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    # Get form values
    features = [float(x) for x in request.form.values()]

    # Convert into numpy array
    final_input = np.array([features])

    # Make prediction
    prediction = model.predict(final_input)

    return render_template(
        "index.html",
        prediction_text=f"Predicted House Price: {prediction[0] * 100000:.2f} USD"
    )

if __name__ == "__main__":
    print('Staring Flask server....')
    app.run(debug=True)