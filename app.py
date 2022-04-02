import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #print(request.form.values())
    int_features =[]
    for x in request.form.values():
        try:
            x=int(x)
            int_features.append(x)
        except ValueError:
            int_features.append(x)
            
    
    print(int_features)
    df=pd.read_csv('test.csv');
    #print(df);
    #df = pd.DataFrame(columns = ['ID', 'age', 'job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome'])
    #df.loc[len(df.index)] =[99999,32,"services","married","secondary","no",118,"yes","no","cellular",15,"may",20,6,-1,0,"unknown"]
    df.loc[len(df.index)]=int_features
    print(df);
    features=pd.get_dummies(df)
    #print(features)
    row1=features.tail(1)
    #final_features = [np.array(features)]
    prediction = model.predict(row1)
    print(prediction[0])


    return render_template('index.html', prediction_text='Decision $ {}'.format(prediction[0]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)