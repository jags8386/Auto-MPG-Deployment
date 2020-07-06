import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("Auto-mpg-lr-model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    temp_array = list()
    cyl = int(request.form["cylinders"])
    disp = float(request.form["displacement"])
    hp = float(request.form["horsepower"])
    wt = float(request.form["weight"])
    acc = float(request.form["acceleration"])
    yr = int(int(request.form["acceleration"]) % 100)
    origin = int(request.form["origin"])

    temp_array = temp_array + [cyl, disp, hp, wt, acc, yr, origin]

    data = np.array([temp_array])

    prediction = model.predict(data)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Car milage should be {}'.format(-1*output))

if __name__ == '__main__':
	app.run(debug=True)