from flask import Flask
from flask import render_template
from flask import request
import statsmodels
import numpy as np
import pickle

app = Flask(__name__, template_folder='templates')

OLS_model = pickle.load(open('model/model_regressor.pkl','rb'))

@app.route('/')
def index():
    return(render_template('main.html'))

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    y_pred_OLS = OLS_model.predict(final_features)
    hasil = []
    return render_template('main.html',prediction_text=hasil[y_pred_OLS[0]])