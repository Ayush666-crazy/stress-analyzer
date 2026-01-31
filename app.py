from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("stress_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    stress_label = None
    stress_score = None
    suggestion = None

    if request.method == "POST":
        
        role = int(request.form["role"])
        screen = int(request.form["screen"])
        exercise = int(request.form["exercise"])
        lifestyle = int(request.form["lifestyle"])

    
        result = model.predict([[role, screen, exercise, lifestyle]])[0]

        
        if result == 0:
            stress_label = "Low Stress 😌"
            stress_score = 25
            suggestion = "You're doing great! Keep a balanced routine and stay active 💪"
        elif result == 1:
            stress_label = "Medium Stress 😐"
            stress_score = 55
            suggestion = "Try reducing screen time, take short breaks, and go for walks 🚶"
        else:
            stress_label = "High Stress 😵"
            stress_score = 85
            suggestion = "Please rest well, sleep properly, exercise lightly, and reduce scrolling 🛌"

    return render_template(
        "index.html",
        stress=stress_label,
        score=stress_score,
        tip=suggestion
    )


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


