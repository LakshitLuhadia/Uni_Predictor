from flask import Flask, request, render_template
from joblib import load

app = Flask(__name__, template_folder="templates")
app.config['TEMPLATES_AUTO_RELOAD'] = True

model = load("admission_predictor_model.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        gre = request.form.get("gre", type=int)
        toefl = request.form.get("toefl", type=int)
        uni_rating = request.form.get("uni_rating", type=int)
        cgpa = request.form.get("cgpa", type=int)
        research = request.form.get("research", type=int)

        prediction = model.predict([[gre, toefl, uni_rating, 3, 3, cgpa, research, 0]])[0]
        return render_template("index.html", output=(prediction+1)*100)
        
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
