# Creaating a Flask application
from flask import Flask, request, render_template
import pickle
import pandas as pd

# Create a flask app
application = Flask(__name__)
app = application

# Creating Homepage path
@app.route('/')
def home_page():
    return render_template('index.html')

# Creating prediction path
@app.route('/predict',methods=['POST'])
def predict_point():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        with open('notebooks/model.pkl','rb') as f:
            model = pickle.load(f)
        sepal_length = float(request.form.get('sepal_length'))
        sepal_width = float(request.form.get('sepal_width'))
        petal_length = float(request.form.get('petal_length'))
        petal_width = float(request.form.get('petal_width'))
        df = pd.DataFrame([sepal_length,sepal_width,petal_length,petal_width]).T
        df.columns = ['sepal_length','sepal_width','petal_length','petal_width']

        pred = model.predict(df)
        pred = pred[0]

        if pred==0:
            prediction = 'Setosa'
        elif pred==1:
            prediction = 'Versicolor'
        else:
            prediction = 'Virginica'

    return render_template('index.html',prediction=prediction)

# Running the app
if __name__ == '__main__':
    app.run(host='0.0.0.0')