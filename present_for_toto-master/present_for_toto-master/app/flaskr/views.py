from flask import (
    Flask,
    jsonify,
    render_template, 
    request
)


from . import models
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/_type_in_num')
def type_in_num():
    tweet = request.args.get('num', 0, type=str)
    result = models.SentimentAnalysis.predict(tweet)
    return jsonify({
        'result':result
    })

    
