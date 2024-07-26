from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('fish_species_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if data is None:
            return jsonify({'error': 'No input data provided'}), 400

        required_fields = ['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400

        input_data = pd.DataFrame([[data[field] for field in required_fields]], columns=required_fields)
        prediction = model.predict(input_data)
        species = label_encoder.inverse_transform(prediction)[0]

        return jsonify({'species': species})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)