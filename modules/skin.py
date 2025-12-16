import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config.landmark_indices import FACE_OVAL, LIPS_LANDMARKS, EYES_LANDMARKS, BROWS_LANDMARKS

def get_mask_from_points(image_shape, points, scale=1.0):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if not points:
        return mask
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask

def process_skin(image, landmarks):
    """
    Applique un lissage de peau (Skin Smoothing) pour féminiser le visage.
    Technique: Filtre Bilateral sur la zone du visage, en excluant les yeux et la bouche.
    """
    
    # 1. Création du masque global du visage
    # Indices pour le contour du visage
    face_oval_points = [landmarks[i] for i in FACE_OVAL]
    
    full_face_mask = get_mask_from_points(image.shape, face_oval_points)
    
    # 2. Création des masques d'exclusion (Yeux, Bouche, Sourcils)
    # On ne veut pas flouter ces zones
    
    # Yeux
    left_eye_pts = [landmarks[i] for i in EYES_LANDMARKS['left']]
    right_eye_pts = [landmarks[i] for i in EYES_LANDMARKS['right']]
    
    # Bouche
    lips_pts = [landmarks[i] for i in LIPS_LANDMARKS['outer']]
    
    # Sourcils (Optionnel, mieux vaut garder du détail)
    left_brow_pts = [landmarks[i] for i in BROWS_LANDMARKS['left']]
    right_brow_pts = [landmarks[i] for i in BROWS_LANDMARKS['right']]

    # Dessiner les exclusions en NOIR (0) sur le masque du visage
    # On dilate un peu les exclusions pour être sûr de ne pas baver sur les bords des yeux/bouche
    exclusion_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    cv2.fillPoly(exclusion_mask, [np.array(left_eye_pts, np.int32)], 255)
    cv2.fillPoly(exclusion_mask, [np.array(right_eye_pts, np.int32)], 255)
    cv2.fillPoly(exclusion_mask, [np.array(lips_pts, np.int32)], 255)
    cv2.fillPoly(exclusion_mask, [np.array(left_brow_pts, np.int32)], 255)
    cv2.fillPoly(exclusion_mask, [np.array(right_brow_pts, np.int32)], 255)
    
    # Dilater les zones d'exclusion (Yeux/Bouche) pour marge de sécurité
    kernel = np.ones((5,5), np.uint8)
    exclusion_mask = cv2.dilate(exclusion_mask, kernel, iterations=2)
    
    # Masque Final = Visage AND NOT Exclusions
    # full_face_mask est uint8 (0 ou 255)
    # On retire l'exclusion
    final_skin_mask = cv2.bitwise_and(full_face_mask, cv2.bitwise_not(exclusion_mask))
    
    # Adoucir les bords du masque pour éviter les coupures nettes
    final_skin_mask_blurred = cv2.GaussianBlur(final_skin_mask, (15, 15), 0)
    
    # 3. Application du filtre (Lissage)
    # Bilateral Filter est excellent pour lisser la peau tout en gardant les edges (bords du visage)
    # Paramètres : d=9, sigmaColor=75, sigmaSpace=75 (Valeurs classiques pour peau)
    # Pour un effet M2F (effacer barbe), on peut forcer un peu.
    blurred_image = cv2.bilateralFilter(image, d=15, sigmaColor=80, sigmaSpace=80)
    
    # Pour un effet encore plus "soft glim", on peut mixer avec un GaussianBlur léger
    # blurred_image = cv2.addWeighted(blurred_image, 0.7, cv2.GaussianBlur(image, (5,5), 0), 0.3, 0)

    # 4. Compositing
    # On combine l'image originale et l'image floutée en utilisant le masque
    
    # Convertir le masque 1 channel -> 3 channels float pour multiplication
    mask_3c = cv2.cvtColor(final_skin_mask_blurred, cv2.COLOR_GRAY2BGR) / 255.0
    
    image_float = image.astype(float)
    blurred_float = blurred_image.astype(float)
    
    # Resultat = Blurred * Mask + Original * (1 - Mask)
    result = blurred_float * mask_3c + image_float * (1.0 - mask_3c)
    
    return result.astype(np.uint8)


if __name__ == "__main__":
    from utils.landmarks import FaceLandmarks
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, "male_face_test.png")
    
    if not os.path.exists(image_path):
        print(f"Erreur: Image introuvable à {image_path}")
    else:
        print(f"Chargement de {image_path}...")
        image = cv2.imread(image_path)
        fl = FaceLandmarks()
        landmarks = fl.get_landmarks(image)
        
        if landmarks:
            print("Landmarks détectés. Application du lissage peau...")
            res = process_skin(image, landmarks)
            out_path = os.path.join(base_dir, "result_skin.jpg")
            cv2.imwrite(out_path, res)
            print(f"Sauvegardé : {out_path}")
        else:
            print("Pas de landmarks.")
