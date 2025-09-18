import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'autism_predictor_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)

model = None
scaler = None
feature_names = None

GENDER_MAP = {'m': 0, 'f': 1}
JAUNDICE_MAP = {'no': 0, 'yes': 1}
AUSTIM_MAP = {'no': 0, 'yes': 1}
USED_APP_BEFORE_MAP = {'no': 0, 'yes': 1}
AGE_DESC_MAP = {'18 and more': 0, '17 and less': 1}

ALL_ETHNICITIES = ['White-European', 'Middle Eastern', 'Asian', 'Black', 'South Asian',
                   'Pasifika', 'Others', 'Latino', 'Hispanic', 'Turkish']
ALL_RELATIONS = ['Self', 'Parent', 'Relative', 'Others', 'Health care professional']

def load_model_and_scaler():
    global model, scaler, feature_names
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        print("Model, scaler, and feature names loaded successfully.")
    except FileNotFoundError:
        print(f"Error: One or more files not found.")
        model = None
        scaler = None
        feature_names = None
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        scaler = None
        feature_names = None

load_model_and_scaler()

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or feature_names is None:
        return jsonify({"error": "Model, scaler, or features not loaded."}), 500

    try:
        data = request.get_json(force=True)
        a_scores = [data.get(f'A{i}_Score', 0) for i in range(1, 11)]
        age = data.get('age', 0)
        gender = data.get('gender', 'm')
        ethnicity = data.get('ethnicity', 'Others')
        jaundice = data.get('jaundice', 'no')
        austim = data.get('austim', 'no')
        country_of_res = data.get('country_of_res', 'United States')
        used_app_before = data.get('used_app_before', 'no')
        relation = data.get('relation', 'Self')

        input_df = pd.DataFrame([[*a_scores, age, gender, ethnicity, jaundice, austim,
                                  country_of_res, used_app_before, relation]],
                                columns=[f'A{i}_Score' for i in range(1, 11)] + 
                                ['age', 'gender', 'ethnicity', 'jaundice', 'austim',
                                 'contry_of_res', 'used_app_before', 'relation'])

        # Clean/standardize
        for col in ['ethnicity', 'relation']:
            val = str(input_df[col].iloc[0]).strip()
            if val == '?' or val.lower() == 'others':
                input_df[col] = 'Others'
            elif val.lower() == 'middle eastern':
                input_df[col] = 'Middle Eastern'
            else:
                input_df[col] = val

        input_df['gender'] = input_df['gender'].map(GENDER_MAP)
        input_df['jaundice'] = input_df['jaundice'].map(JAUNDICE_MAP)
        input_df['austim'] = input_df['austim'].map(AUSTIM_MAP)
        input_df['used_app_before'] = input_df['used_app_before'].map(USED_APP_BEFORE_MAP)
        input_df['age_desc'] = input_df['age'].apply(lambda x: '18 and more' if x >= 18 else '17 and less')
        input_df['age_desc'] = input_df['age_desc'].map(AGE_DESC_MAP)

        # One-hot encoding
        ethnicity_dummies = pd.get_dummies(input_df['ethnicity'], prefix='ethnicity')
        relation_dummies = pd.get_dummies(input_df['relation'], prefix='relation')
        ethnicity_dummies = ethnicity_dummies.reindex(columns=[f'ethnicity_{e}' for e in ALL_ETHNICITIES], fill_value=0)
        relation_dummies = relation_dummies.reindex(columns=[f'relation_{r}' for r in ALL_RELATIONS], fill_value=0)

        input_df = input_df.drop(columns=['ethnicity', 'relation', 'contry_of_res', 'age_desc'], errors='ignore')
        input_df = pd.concat([input_df, ethnicity_dummies, relation_dummies], axis=1)

        # Reindex to model's feature list
        input_data_aligned = input_df.reindex(columns=feature_names, fill_value=0)

        # Scale numeric features
        numerical_features = [f'A{i}_Score' for i in range(1, 11)] + ['age']
        input_data_aligned[numerical_features] = scaler.transform(input_data_aligned[numerical_features])

        # Prediction
        prediction = model.predict(input_data_aligned)[0]
        probability = model.predict_proba(input_data_aligned)[0].tolist()

        return jsonify({
            "prediction": int(prediction),
            "probability": probability
        })

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
