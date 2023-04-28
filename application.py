from flask import Flask,request,render_template
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application = Flask(__name__)

app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')

    else:
        data = CustomData(
            Delivery_person_Age = int(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings = float(request.form.get('Delivery_person_Ratings')),
            Weather_conditions = request.form.get('Weather_conditions'),
            Road_traffic_density = request.form.get('Road_traffic_density'),
            Vehicle_condition =  int(request.form.get('Vehicle_condition')),
            multiple_deliveries = int(request.form.get('multiple_deliveries')),
            Dist_from_Rest_to_deli_loc = float(request.form.get('Dist_from_Rest_to_deli_loc'))
        )
        
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)
        
        result = int(pred[0])
        
        return render_template('form.html',final_result=result)
        
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
