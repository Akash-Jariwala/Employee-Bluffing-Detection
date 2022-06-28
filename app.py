import pickle
from pyexpat import model
from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_reg = PolynomialFeatures(degree=3)

app = Flask(__name__)

model = pickle.load(open('bluffingDetectionModel.pkl','rb'))

dataset = pd.read_csv('Position_Salaries.csv')

@app.route('/')
def index():
    levels = sorted(dataset['Level'].unique())
    return render_template('index2.html',levels=levels,tables=[dataset.to_html()], titles=[''])

@app.route('/predict',methods=["POST"])
def predict():
    level = request.form.get('level')

    prediction = model.predict(poly_reg.fit_transform(pd.DataFrame([[level]], columns=['Level'])))

    return str(np.round(prediction[0],2))

if __name__ == "__main__":
    app.run(debug=True)
