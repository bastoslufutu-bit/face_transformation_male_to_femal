import os
import sys
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import time

# Configuration du Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports Modules
from utils.landmarks import FaceLandmarks
from modules.nose import process_nose
from modules.skin import process_skin
from modules.lips import process_lips
from modules.brows import process_brows
from modules.hair import process_hair
from modules.beard import process_beard
from modules.eyes import process_eyes
from modules.jaw import process_jaw
from modules.cheeks import process_cheeks
from modules.chin import process_chin

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max

# Création des dossiers
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Initialisation Landmarks
fl = FaceLandmarks()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transform', methods=['POST'])
def transform():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image envoyée'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Pas de fichier sélectionné'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        # Timestamp pour éviter le cache
        timestamp = int(time.time())
        filename_base, ext = os.path.splitext(filename)
        new_filename = f"{filename_base}_{timestamp}{ext}"
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        file.save(filepath)
        
        # --- PIPELINE DE TRANSFORMATION ---
        try:
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'Impossible de lire l\'image'}), 400
                
            landmarks = fl.get_landmarks(image)
            if not landmarks:
                return jsonify({'error': 'Aucun visage détecté !'}), 400
                
            # Ordre Optimal pour M2F
            
            # 1. Structure (On sculpte le visage avant de peindre)
            img_step = process_nose(image, landmarks)
            img_step = process_jaw(img_step, landmarks)
            img_step = process_chin(img_step, landmarks)
            img_step = process_cheeks(img_step, landmarks)
            
            # 2. Nettoyage (Barbe + Peau)
            img_step = process_beard(img_step, landmarks) # Enlever la barbe
            img_step = process_skin(img_step, landmarks)  # Lisser la peau
            
            # 3. Features (Yeux, Sourcils, Lèvres)
            img_step = process_eyes(img_step, landmarks)
            img_step = process_brows(img_step, landmarks)
            img_step = process_lips(img_step, landmarks)
            
            # 4. Cheveux (En dernier pour recouvrir le tout)
            final_result = process_hair(img_step, landmarks)
            
            # Sauvegarder le résultat
            result_filename = f"result_{new_filename}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            cv2.imwrite(result_path, final_result)
            
            return jsonify({
                'original_url': f"/{filepath.replace(os.path.sep, '/')}",
                'result_url': f"/{result_path.replace(os.path.sep, '/')}"
            })

        except Exception as e:
            print(f"Erreur Transformation: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Utilisation du port 5050 et host 0.0.0.0 pour éviter les conflits et écouter partout
    print("Démarrage du serveur sur http://127.0.0.1:5050...")
    app.run(debug=True, port=5050, host='0.0.0.0')
