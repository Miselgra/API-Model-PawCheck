from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model
MODEL_PATH = "Models/ModelDogDisease.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception(f"Model file tidak ditemukan di path: {MODEL_PATH}")

# Daftar fitur
FEATURE_NAMES = [
    "Jenis Hewan",           
    "Umur",                  
    "Hari sakit",            
    "Kehilangan nafsu makan",  
    "Muntah",                 
    "Diare berdarah",         
    "Batuk",                  
    "Sesak napas",            
    "Lemas",                  
    "luka pada kulit",        
    "Cairan hidung",          
    "Cairan mata"             
]

NUMERIC_FEATURES = {"Jenis Hewan", "Umur", "Hari sakit"}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if isinstance(data, dict):
            data = [data]

        predictions = []
        for entry in data:
            if not all(key in entry for key in FEATURE_NAMES):
                return jsonify({
                    "error": f"Setiap input harus berisi fitur berikut: {FEATURE_NAMES}"
                }), 400

            features = []
            feature_map = {}
            for key in FEATURE_NAMES:
                if key in NUMERIC_FEATURES:
                    try:
                        value = float(entry[key])
                        features.append(value)
                        feature_map[key] = value
                    except ValueError:
                        return jsonify({"error": f"Nilai untuk '{key}' harus numerik."}), 400
                else:
                    value = str(entry[key]).strip().lower()
                    if value == "yes":
                        features.append(1)
                        feature_map[key] = 1
                    elif value == "no":
                        features.append(0)
                        feature_map[key] = 0
                    else:
                        return jsonify({
                            "error": f"Nilai untuk '{key}' harus 'yes' atau 'no'."
                        }), 400

            # ----- Prediksi dari aturan statis -----
            prediction_by_rule = "Tidak diketahui"
            probabilities_by_rule = [0.33, 0.33, 0.34]  # default

            if (
                feature_map["Kehilangan nafsu makan"] == 1 and
                feature_map["Muntah"] == 1 and
                feature_map["Diare berdarah"] == 1 and
                feature_map["Batuk"] == 1
            ):
                prediction_by_rule = "Canine Distemper"
                probabilities_by_rule = [0.948, 0.025, 0.027]
            elif feature_map["Kehilangan nafsu makan"] == 1 and feature_map["Muntah"] == 1:
                prediction_by_rule = "Canine Leptospirosis"
                probabilities_by_rule = [0.015, 0.955, 0.03]
            elif feature_map["Muntah"] == 1 and feature_map["Diare berdarah"] == 1:
                prediction_by_rule = "Canine Parvovirus"
                probabilities_by_rule = [0.025, 0.03, 0.945]

            # ----- Prediksi dari model -----
            input_df = pd.DataFrame([features], columns=FEATURE_NAMES)
            prediction_by_model = model.predict(input_df)[0]
            probabilities_by_model = model.predict_proba(input_df)[0].tolist()

            # ----- Bandingkan probabilitas tertinggi -----
            max_rule = max(probabilities_by_rule)
            max_model = max(probabilities_by_model)

            if max_rule >= max_model:
                final_prediction = prediction_by_rule
                final_probabilities = probabilities_by_rule
                source = "rule"
            else:
                final_prediction = prediction_by_model
                final_probabilities = probabilities_by_model
                source = "model"

            # ----- Gabungkan hasil -----
            predictions.append({
                "prediction_by_rule": prediction_by_rule,
                "probabilities_by_rule": probabilities_by_rule,
                "prediction_by_model": prediction_by_model,
                "probabilities_by_model": probabilities_by_model,
                "prediction": final_prediction,
                "probabilities": final_probabilities,
                "final_source": source  # tambahan info: asal prediksi final
            })

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)