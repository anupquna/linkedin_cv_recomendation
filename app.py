import numpy as np
from flask import Flask,escape, request, jsonify, render_template
import pickle

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

	
@app.route('/test_predict',methods=['POST'])
def test_predict():
	int_features = [x for x in request.form.values()]
	final_features = [np.array(int_features)]
	return render_template("result.html", user=final_features)

if __name__ == "__main__":
    app.run(debug=True)