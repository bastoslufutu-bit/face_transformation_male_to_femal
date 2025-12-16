import cv2
import os
import sys
import numpy as np

# Ajouter le path du projet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.landmarks import FaceLandmarks
from modules.brows import process_brows

def test():
    print("Début du test Brows...")
    
    # Charger image
    image_path = "male_face_test.png"
    if not os.path.exists(image_path):
        print(f"Image {image_path} introuvable.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Echec chargement image.")
        return
    
    print(f"Image chargée: Shape {image.shape}")

    # Landmarks
    fl = FaceLandmarks()
    landmarks = fl.get_landmarks(image)
    
    if not landmarks:
        print("Pas de landmarks.")
        return
    
    print(f"Landmarks détectés: {len(landmarks)} points.")
    
    # Process
    try:
        print("Appel de process_brows...")
        result = process_brows(image.copy(), landmarks)
        print("process_brows terminé succès.")
        
        cv2.imwrite("result_brows_debug.jpg", result)
        print("Resultat sauvegardé.")
        
    except Exception as e:
        print(f"EXCEPTION Python attrapée: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
