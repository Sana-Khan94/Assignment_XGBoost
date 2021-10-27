from flask import Flask, render_template, request
from flask import Response
from flask_cors import cross_origin
import pickle
import pandas as pd

app = Flask(__name__)


@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            # age	workclass	fnlwgt	education_num	marital_status	occupation	relationship	race
            # sex	capital_gain	capital_loss	hours_per_week	native_country
            age = float(request.form['age'])
            workclass = float(request.form['workclass'])
            fnlwgt = float(request.form['fnlwgt'])
            education_num = float(request.form['education_num'])
            marital_status = float(request.form['marital_status'])
            occupation = float(request.form['occupation'])
            relationship = float(request.form['relationship'])
            race = float(request.form['race'])
            sex = float(request.form['sex'])
            capital_gain = float(request.form['capital_gain'])
            capital_loss = float(request.form['capital_loss'])
            hours_per_week = float(request.form['hours_per_week'])
            native_country = float(request.form['native_country'])
            prediction = predict_xgb([[age, workclass, fnlwgt, education_num, marital_status, occupation,
                                       relationship, race, sex, capital_gain, capital_loss, hours_per_week,
                                       native_country]])
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('results.html', prediction=prediction)
        except Exception as e:
            print('The Exception message is: ', e)
            return e
    # return render_template('results.html')
    else:
        return render_template('index.html')


@app.route("/price_via_postman1", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        if request.json['data'] is not None:
            data = request.json['data']
            print('data is:     ', data)
            res = predict_xgb(data)
            print('result is        ', res)
            return Response(res)
    except ValueError:
        return Response("Value not found")
    except Exception as e:
        print('exception is   ', e)
        return Response(e)


def predict_xgb(dict_pred):
    if request.method == 'POST':
        with open('scaler_model.pickle', 'rb') as f:
            scalar = pickle.load(f)

        with open('xgboost_model.pickle', 'rb') as f:
            model = pickle.load(f)

        data_df = pd.DataFrame(dict_pred, index=[1, ])
        scalar_data = scalar.transform(data_df)
        predict = model.predict(scalar_data)

        if predict[0] == 1:
            result = "Person makes over 50K per year"
        else:
            result = "Person not able to make over 50K per year"

        return result


if __name__ == "__main__":
    host = '0.0.0.0'
    port = 5000
    app.run(debug=False)
