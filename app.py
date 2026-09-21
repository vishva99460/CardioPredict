
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("best_cardiovascular_model.pkl")
# Scaler not required for Random Forest


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        features = [
            float(request.form["gender"]),
            float(request.form["height"]),
            float(request.form["weight"]),
            float(request.form["ap_hi"]),
            float(request.form["ap_lo"]),
            float(request.form["cholesterol"]),
            float(request.form["gluc"]),
            float(request.form["smoke"]),
            float(request.form["alco"]),
            float(request.form["active"]),
            float(request.form["age_years"])
        ]

        data = np.array(features).reshape(1, -1)

        result = model.predict(data)[0]

        if result == 1:
            prediction = "Higher predicted cardiovascular disease risk"
        else:
            prediction = "Lower predicted cardiovascular disease risk"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
