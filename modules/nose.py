import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config.landmark_indices import NOSE_LANDMARKS
from utils.geometry import get_roi_from_landmarks

def process_nose(image, landmarks):
    """
    Transforme le nez pour le rendre plus féminin :
    1. Extraction de la zone du nez.
    2. Affinement (réduction de largeur).
    3. Réintégration.
    """
    indices = NOSE_LANDMARKS['bridge'] + NOSE_LANDMARKS['nostrils'] + NOSE_LANDMARKS['tip']
    
    # 1. Extraction (Cut)
    # On prend une box un peu plus large pour inclure tout le nez
    roi, (x, y, w, h) = get_roi_from_landmarks(image, landmarks, indices, padding=20)
    
    if roi.size == 0:
        return image
        
    # 2. Transformation (Make it female)
    # Caractéristiques féminines : Nez plus fin, moins large.
    # On va utiliser un resize pour compresser la largeur (width)
    
    # Facteur de réduction (ex: 85% de la largeur originale)
    shrink_factor = 0.85 
    new_width = int(w * shrink_factor)
    
    # Redimensionnement du ROI (Affinement)
    # On resize toute la ROI, ce qui va "écraser" le nez horizontalement
    transformed_roi = cv2.resize(roi, (new_width, h), interpolation=cv2.INTER_LANCZOS4)
    
    # Pour remettre en place, il faut compenser la perte de largeur.
    # Si on colle une image plus petite, on aura des trous.
    # L'approche "Cut -> Transform -> Paste" stricte laisse des vides si on réduit. 
    # Pour éviter les trous noirs, on doit redimensionner le transformed_roi pour qu'il remplisse
    # la zone centrale, mais le mieux est de faire un "Seam Carving" ou un Warping local.
    
    # APPROCHE UTILISATEUR "Remove -> Transform -> Put back" 
    # Pour que ça marche sans trous, on va centrer le nez affiné dans un fond qui matche la peau
    # ou utiliser Poisson Blending (seamlessClone).
    
    # Créons un masque pour le seamlessClone est la meilleure option pour "remettre à la place"
    # Mais seamlessClone a besoin que la source et la destination aient la meme taille ou on colle une petite source.
    
    # On va centrer le nez réduit au milieu de la boite originale
    result_roi = roi.copy()
    
    # Coordonnees pour centrer
    start_x = (w - new_width) // 2
    
    # On "colle" le nez affiné au milieu du ROI original. 
    # Attention: les bords du nez affiné vont se voir. 
    # On applique un léger flou sur les bords verticaux du nez affiné avant de coller ?
    # Ou mieux : Seamless Clone du nez affiné SUR la zone originale.
    
    mask = 255 * np.ones(transformed_roi.shape, transformed_roi.dtype)
    
    # Centre de la zone de destination
    center = (x + w//2, y + h//2)
    
    # Seamless Clone 
    # Note: Opencv seamlessClone prend (src, dst, mask, center)
    # src = transformed_roi (le nez fin)
    # dst = image (l'image originale complet)
    try:
        image = cv2.seamlessClone(transformed_roi, image, mask, center, cv2.NORMAL_CLONE)
    except Exception as e:
        print(f"Error in nose transformation: {e}")
        # Fallback si ça echoue (ex: bordure d'image)
        pass

    return image

if __name__ == "__main__":
    from utils.landmarks import FaceLandmarks
    
    # Chemin vers l'image de test test
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, "male_face_test.png")
    
    if not os.path.exists(image_path):
        print(f"Erreur: Image introuvable à {image_path}")
    else:
        print(f"Chargement de {image_path}...")
        image = cv2.imread(image_path)
        
        if image is None:
            print("Erreur lors du chargement de l'image.")
        else:
            fl = FaceLandmarks()
            landmarks = fl.get_landmarks(image)
            
            if landmarks:
                print("Landmarks détectés.")
                result = process_nose(image.copy(), landmarks)
                
                output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "result_nose.jpg")
                cv2.imwrite(output_path, result)
                print(f"Transformation terminée. Résultat sauvegardé dans : {output_path}")
            else:
                print("Aucun visage détecté.")
