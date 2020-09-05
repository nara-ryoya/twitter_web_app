from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.metrics import classification_report, accuracy_score
# import pickle
# データ取得
iris = load_iris()
x, y = iris.data, iris.target
# 訓練データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
# モデルのインスタンスを作成(ニューラルネットワーク)
model = MLPClassifier(solver="sgd", random_state=0, max_iter=3000)
# 学習
model.fit(x_train, y_train)
pred = model.predict(x_test)
# 学習済みモデルの保存
joblib.dump(model, "nn.pkl", compress=True)
# 予測精度
print("result: ", model.score(x_test, y_test))
print(classification_report(y_test, pred))
app.py
from flask import Flask, render_template, request
from wtforms import Form, FloatField, SubmitField, validators, ValidationError
import numpy as np
import joblib
# 学習モデルを読み込み予測する
def predict(parameters):
    # モデル読み込み
    model = joblib.load('./nn.pkl')
    params = parameters.reshape(1,-1)
    pred = model.predict(params)
    return pred
# ラベルからIrisの名前を取得
def getName(label):
    print(label)
    if label == 0:
        return "Iris Setosa"
    elif label == 1: 
        return "Iris Versicolor"
    elif label == 2: 
        return "Iris Virginica"
    else: 
        return "Error"
app = Flask(__name__)
# Flaskとwtformsを使い、index.html側で表示させるフォームを構築する
class IrisForm(Form):
    SepalLength = FloatField("Sepal Length(cm)（蕚の長さ）",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])
    SepalWidth  = FloatField("Sepal Width(cm)（蕚の幅）",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])
    PetalLength = FloatField("Petal length(cm)（花弁の長さ）",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])
    PetalWidth  = FloatField("petal Width(cm)（花弁の幅）",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])
    # html側で表示するsubmitボタンの表示
    submit = SubmitField("判定")
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    form = IrisForm(request.form)
    if request.method == 'POST':
        if form.validate() == False:
            return render_template('index.html', form=form)
        else:            
            SepalLength = float(request.form["SepalLength"])            
            SepalWidth  = float(request.form["SepalWidth"])            
            PetalLength = float(request.form["PetalLength"])            
            PetalWidth  = float(request.form["PetalWidth"])
            x = np.array([SepalLength, SepalWidth, PetalLength, PetalWidth])
            pred = predict(x)
            irisName = getName(pred)
            return render_template('result.html', irisName=irisName)
    elif request.method == 'GET':
        return render_template('index.html', form=form)
if __name__ == "__main__":
    app.run()
index.html
<title>Iris Predict App</title>
<style>
    #wrapper {
        text-align: center;
    }
</style>
<div id="wrapper">
    <form method="post">
        {{ form.SepalLength.label }}<br>
        {{ form.SepalLength }}
        <br>
        {{ form.SepalWidth.label }}<br>
        {{ form.SepalWidth }}
        <br>
        {{ form.PetalLength.label }}<br>
        {{ form.PetalLength }}
        <br>
        {{ form.PetalWidth.label }}<br>
        {{ form.PetalWidth }}
        <br>
        {{ form.submit }}
    </form>
</div>