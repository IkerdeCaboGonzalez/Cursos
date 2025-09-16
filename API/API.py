from importlib.resources import contents
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib 

def return_prediction(model, scaler, sample_json):
  s_len=sample_json['sepal_length']
  s_wid=sample_json['sepal_width']
  p_len=sample_json['petal_length']
  p_wid=sample_json['petal_width']

  flower=[[s_len, s_wid, p_len, p_wid]]

  classes=np.array(['setosa', 'versicolor', 'virginica'])

  flower=scaler.transform(flower)

  class_ind=model.predict(flower)
  class_ind=classes[np.argmax(class_ind)]

  return class_ind


app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Flask API is running!</h1>'

model_pretrained = load_model('iris_model.h5')
scaler_prefitted = joblib.load('iris_scaler.pkl')

@app.route('/api/flower', methods=['POST'])
def flower_prediction():
   content = request.json
   result = return_prediction(model_pretrained, scaler_prefitted, content)
   return jsonify(result)

if __name__ == '__main__':
    app.run()



