from flask import Flask, request,render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import Predictpipeline,CustomData


application = Flask(__name__)

app = application

# Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            fixed_acidity = float(request.form.get('fixed_acidity')),
            volatile_acidity = float(request.form.get('volatile_acidity')),
            citric_acid = float(request.form.get('citric_acid')),
            residual_sugar = float(request.form.get('residual_sugar')),
            chlorides = float(request.form.get('chlorides')),
            free_sulfur_dioxide = float(request.form.get('free_sulfur_dioxide')),
            total_sulfur_dioxide = float(request.form.get('total_sulfur_dioxide')),
            density = float(request.form.get('density')),
            pH = float(request.form.get('pH')),
            sulphates = float(request.form.get('sulphates')),
            alcohol = float(request.form.get('alcohol'))
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = Predictpipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)