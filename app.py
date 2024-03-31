from flask import Flask, request, render_template
from joblib import load

app = Flask(__name__, template_folder="templates")
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load your model (adjust the path as necessary)
model = load("admission_predictor_model.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Extract the input data from the form
        gre = request.form.get("gre", type=int)
        toefl = request.form.get("toefl", type=int)
        uni_rating = request.form.get("uni_rating", type=int)
        sop = request.form.get("sop", type=int)
        lor = request.form.get("lor", type=int)
        cgpa = request.form.get("cgpa", type=int)
        research = request.form.get("research", type=int)
        # halwa = request.form.get("halwa", type=int)

        # Add additional inputs as necessary

        # Make a prediction using the model
        # Ensure the input data is in the correct format
        prediction = model.predict([[gre, toefl, uni_rating, sop, lor, cgpa, research, 0]])[0]

        # Return a rendered template with the prediction result
        return render_template("index.html", output=(prediction+1)*100)
    else:
        # No form data submitted, render the form
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
