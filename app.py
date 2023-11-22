from flask import Flask, render_template, request
import numpy as np
import pickle
#import joblib
app = Flask(__name__)
filename = 'file_heat.pkl'
model = pickle.load(open(filename, 'rb'))    # load the model
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # The user input is processed here
def predict():
    SEX = request.form['SEX']
    BRANCH = request.form['BRANCH']
    RACE_CODE = request.form['RACE_CODE']
    POSITION = request.form['POSITION']
    AGE_4_GROUP = request.form['AGE_4_GROUP']
    STATE = request.form['STATE']
    pred = model.predict(np.array([[SEX,BRANCH,RACE_CODE,POSITION,AGE_4_GROUP,STATE ]]).astype(int))[0]
    #print(pred)
    return render_template('index.html', predict=str(pred))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


