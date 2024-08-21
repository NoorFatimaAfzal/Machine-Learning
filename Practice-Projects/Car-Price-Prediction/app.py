from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the dataset
df = pd.read_csv(r'C:\Users\InfoBay\OneDrive\Desktop\Machine_Learning\Practice-Projects\Car-Price-Prediction\Cleaned_car.csv')

# Load the model
model = pickle.load(open(r'C:\Users\InfoBay\OneDrive\Desktop\Machine_Learning\Practice-Projects\Car-Price-Prediction\2_LinearRegressionModel.pkl', 'rb'))

@app.route('/')
def index():
    company = sorted(df['company'].unique())
    year = sorted(df['year'].unique(), reverse=True)
    fuel_type = df['fuel_type'].unique()
    return render_template('index.html', companies=company, years=year, fuel_types=fuel_type)

@app.route('/get_models')
def get_models():
    company = request.args.get('company')
    models = sorted(df[df['company'] == company]['name'].unique())
    return jsonify({'models': models})

@app.route('/predict_price')
def predict_price():
    company = request.args.get('company')
    model_name = request.args.get('model')
    year = int(request.args.get('year'))
    kms_driven = int(request.args.get('kmsDriven'))
    fuel_type = request.args.get('fuelType')

    # Prepare the data for prediction
    data = pd.DataFrame([[company, model_name, year, kms_driven, fuel_type]], columns=['company', 'name', 'year', 'kms_driven', 'fuel_type'])

    try:
        # Make prediction
        price = model.predict(data)[0]
        return jsonify({'price': price})
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
