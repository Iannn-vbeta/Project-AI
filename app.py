from flask import Flask, request, jsonify
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)

# Fuzzy Logic Setup
# Input Variables
suhu = ctrl.Antecedent(np.arange(0, 41, 1), 'Suhu')
kelembapan = ctrl.Antecedent(np.arange(0, 101, 1), 'Kelembapan')
kelelahan = ctrl.Antecedent(np.arange(0, 11, 1), 'Kelelahan')

# Output Variable
waktu_berolahraga = ctrl.Consequent(np.arange(0, 5, 1), 'Waktu_Berolahraga')

# Membership Functions
suhu['Dingin'] = fuzz.trapmf(suhu.universe, [0, 0, 10, 20])
suhu['Normal'] = fuzz.trimf(suhu.universe, [15, 25, 35])
suhu['Panas'] = fuzz.trapmf(suhu.universe, [25, 35, 40, 40])

kelembapan['Rendah'] = fuzz.trimf(kelembapan.universe, [0, 0, 50])
kelembapan['Sedang'] = fuzz.trimf(kelembapan.universe, [30, 50, 70])
kelembapan['Tinggi'] = fuzz.trimf(kelembapan.universe, [50, 100, 100])

kelelahan['Rendah'] = fuzz.trimf(kelelahan.universe, [0, 0, 3])
kelelahan['Sedang'] = fuzz.trimf(kelelahan.universe, [2, 5, 8])
kelelahan['Tinggi'] = fuzz.trimf(kelelahan.universe, [7, 10, 10])

waktu_berolahraga['Pagi'] = fuzz.trimf(waktu_berolahraga.universe, [0, 0, 1])
waktu_berolahraga['Siang'] = fuzz.trimf(waktu_berolahraga.universe, [1, 1, 2])
waktu_berolahraga['Sore'] = fuzz.trimf(waktu_berolahraga.universe, [2, 2, 3])
waktu_berolahraga['Malam'] = fuzz.trimf(waktu_berolahraga.universe, [3, 4, 4])

# Fuzzy Rules
rules = [
    ctrl.Rule(suhu['Dingin'] & kelembapan['Rendah'] & kelelahan['Rendah'], waktu_berolahraga['Pagi']),
    ctrl.Rule(suhu['Normal'] & kelembapan['Sedang'] & kelelahan['Rendah'], waktu_berolahraga['Sore']),
    ctrl.Rule(suhu['Panas'] & kelembapan['Tinggi'] & kelelahan['Rendah'], waktu_berolahraga['Malam']),
    ctrl.Rule(suhu['Normal'] & kelembapan['Rendah'] & kelelahan['Sedang'], waktu_berolahraga['Siang']),
    ctrl.Rule(suhu['Panas'] & kelembapan['Tinggi'] & kelelahan['Sedang'], waktu_berolahraga['Malam']),
    ctrl.Rule(suhu['Normal'] & kelembapan['Sedang'] & kelelahan['Tinggi'], waktu_berolahraga['Malam']),
    ctrl.Rule(suhu['Panas'] & kelembapan['Tinggi'] & kelelahan['Tinggi'], waktu_berolahraga['Malam'])
]

# Control System
waktu_berolahraga_ctrl = ctrl.ControlSystem(rules)

@app.route('/')
def index():
    return "Welcome to the Fuzzy Logic API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        suhu_val = data['suhu']
        kelembapan_val = data['kelembapan']
        kelelahan_val = data['kelelahan']

        # Create a new simulation object for each request
        waktu_berolahraga_sim = ctrl.ControlSystemSimulation(waktu_berolahraga_ctrl)
        waktu_berolahraga_sim.input['Suhu'] = suhu_val
        waktu_berolahraga_sim.input['Kelembapan'] = kelembapan_val
        waktu_berolahraga_sim.input['Kelelahan'] = kelelahan_val
        waktu_berolahraga_sim.compute()

        result = waktu_berolahraga_sim.output['Waktu_Berolahraga']

        # Tentukan kategori berdasarkan nilai fuzzy
        if result <= 1.5:
            kategori = "Pagi"
        elif result <= 2.5:
            kategori = "Siang"
        elif result <= 3.5:
            kategori = "Sore"
        else:
            kategori = "Malam"

        return jsonify({'waktu_berolahraga': result, 'kategori': kategori})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
