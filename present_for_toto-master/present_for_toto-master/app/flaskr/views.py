from flask import (
    Flask,
    jsonify,
    render_template, 
    request
)


from . import models

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/_type_in_num')
def type_in_num():
    num = request.args.get('num', 0, type=int)
    result = models.SentimentAnalysis.isprime(num)
    return jsonify({
        'result':result
    })

    
