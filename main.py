from flask import Flask, render_template, request
from transformers import pipeline
import pickle

app = Flask("__main__")
classifier = pickle.load(open("data\\classifier.pkl", "rb"))
clf = pipeline("sentiment-analysis")
"""@app.route: This is a decorator provided by Flask for defining routes. Decorators modify the behavior of functions or methods they decorate."""
@app.route("/")
def index():
    return render_template("index.html")

"""methods=['POST']: This parameter indicates that the route will only respond to HTTP POST requests. The route is designed to handle data submitted via a form or an API POST request."""
@app.route("/sentanaylsis", methods = ['POST'] )
def sentanaylsis():
    
    if "sentiment-tan" not in request.form:
        return render_template("index.html")
    sentiment_ta = request.form['sentiment-tan']
    sentiment = clf([sentiment_ta])[0]['label']

    return render_template("sentiment.html", sentiment=sentiment)



@app.route("/sentiment")
def sentiment():
    return render_template("sentiment.html", sentiment="")

if __name__ == "__main__":
    app.run(debug=True)
