import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config.landmark_indices import EYES_LANDMARKS
from utils.geometry import get_roi_from_landmarks

def process_eyes(image, landmarks):
    """
    Féminisation des yeux Version 2 (Intense):
    1. Agrandissement significatif (+6%)
    2. Eyeliner "Cat Eye" (étiré vers l'extérieur)
    3. Ajout de Cils (Lashes)
    4. Éclaircissement du regard
    """
    
    # Copie de travail pour le maquillage
    makeup_layer = image.copy()
    
    for eye_side in ["left", "right"]:
        indices = EYES_LANDMARKS[eye_side]
        
        # --- 1. MAQUILLAGE (Eyeliner + Cils) ---
        # On dessine SUR l'image AVANT l'agrandissement pour que le maquillage soit lui aussi agrandi/intégré
        
        eye_points = np.array([landmarks[i] for i in indices], np.int32)
        
        # Séparer paupière supérieure (les points les plus hauts)
        # On prend le min Y et max Y pour trouver le centre vertical
        min_y = np.min(eye_points[:, 1])
        max_y = np.max(eye_points[:, 1])
        mid_y = (min_y + max_y) // 2
        
        # Paupière sup : points au dessus du milieu
        upper_lid = eye_points[eye_points[:, 1] < mid_y + 2] # +2 de tolérance
        # Trier par X
        upper_lid = upper_lid[np.argsort(upper_lid[:, 0])]
        
        if len(upper_lid) > 2:
            # 1.1 Eyeliner épais
            # On lisse la courbe
            curve_pts = upper_lid
            
            # Cat Eye : On étend le dernier point vers l'extérieur et le haut
            last_pt = curve_pts[-1] if eye_side == "left" else curve_pts[0]
            first_pt = curve_pts[0] if eye_side == "left" else curve_pts[-1]
            
            # Le coin extérieur est le point avec le plus grand X pour l'oeil gauche (sur l'image, à droite du visage ?)
            # Attention : Left Eye = Oeil Gauche du sujet (donc à droite sur l'image si face caméra ?)
            # Mediapipe : Left Eye est l'oeil gauche PHYSIQUE du sujet (donc à gauche de l'image si on regarde l'écran et que le sujet nous regarde ?)
            # Vérifions : indices 362... (Left) sont à des coord X plus grandes que 33... (Right).
            # Donc Left Eye Mediapipe = Droite de l'image.
            
            outer_corner = None
            if eye_side == "left": # (Sur l'image à DROITE)
                outer_corner = upper_lid[np.argmax(upper_lid[:, 0])] # Le plus à droite
                # Direction du trait : vers la droite et un peu vers le haut
                cat_eye_pt = (outer_corner[0] + 12, outer_corner[1] - 7)
            else: # Right Eye (Sur l'image à GAUCHE)
                outer_corner = upper_lid[np.argmin(upper_lid[:, 0])] # Le plus à gauche
                # Direction : vers la gauche et haut
                cat_eye_pt = (outer_corner[0] - 12, outer_corner[1] - 7)
            
            # Dessiner le trait principal
            cv2.polylines(makeup_layer, [upper_lid], False, (20, 20, 20), 2, cv2.LINE_AA)
            
            # Dessiner la pointe Cat Eye
            cv2.line(makeup_layer, tuple(outer_corner), cat_eye_pt, (20, 20, 20), 3, cv2.LINE_AA)
            
            # 1.2 Cils (Lashes)
            # Quelques petits traits verticaux sur la partie extérieure de la paupière
            # On prend le tiers extérieur de la paupière
            num_lashes = 5
            
            # On itère sur les points extérieurs
            start_idx = len(upper_lid) // 2
            relevant_pts = upper_lid[start_idx:] if eye_side == "left" else upper_lid[:len(upper_lid)//2]
            
            # On prend un échantillon de points
            if len(relevant_pts) > 0:
                indices_lashes = np.linspace(0, len(relevant_pts)-1, num_lashes, dtype=int)
                for idx in indices_lashes:
                    pt = relevant_pts[idx]
                    # Cils partent vers le haut
                    lash_len = 6
                    # Légère inclinaison
                    dx = 3 if eye_side == "left" else -3
                    dy = -lash_len
                    end_lash = (pt[0] + dx, pt[1] + dy)
                    cv2.line(makeup_layer, tuple(pt), end_lash, (30, 30, 30), 1, cv2.LINE_AA)

    # Réappliquer le maquillage sur l'image de base
    # On mixe pour que ce ne soit pas trop "paint"
    image = cv2.addWeighted(image, 0.6, makeup_layer, 0.4, 0)


    # --- 2. AGRANDISSEMENT (Zoom) ---
    res_image = image.copy()
    
    for eye_side in ["left", "right"]:
        indices = EYES_LANDMARKS[eye_side]
        roi, (x, y, w, h) = get_roi_from_landmarks(res_image, landmarks, indices, padding=12)
        
        if roi.size == 0:
            continue
            
        # Facteur d'agrandissement FEMININ (+8%)
        scale_factor = 1.08
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        enlarged_eye = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Masque elliptique
        mask = np.zeros((new_h, new_w), dtype=np.float32)
        cv2.ellipse(mask, (new_w//2, new_h//2), (new_w//2, new_h//2), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 5) # Flou moins fort pour garder netteté oeil
        mask = cv2.merge([mask] * 3)
        
        # Collage centrée
        center_x = x + w // 2
        center_y = y + h // 2
        
        dst_tl_x = center_x - new_w // 2
        dst_tl_y = center_y - new_h // 2
        dst_br_x = dst_tl_x + new_w
        dst_br_y = dst_tl_y + new_h
        
        img_h, img_w = res_image.shape[:2]
        dst_x_start = max(0, dst_tl_x)
        dst_y_start = max(0, dst_tl_y)
        dst_x_end = min(img_w, dst_br_x)
        dst_y_end = min(img_h, dst_br_y)
        
        if dst_x_start >= dst_x_end or dst_y_start >= dst_y_end:
            continue
            
        src_x_start = dst_x_start - dst_tl_x
        src_y_start = dst_y_start - dst_tl_y
        src_x_end = src_x_start + (dst_x_end - dst_x_start)
        src_y_end = src_y_start + (dst_y_end - dst_y_start)
        
        src_valid = enlarged_eye[src_y_start:src_y_end, src_x_start:src_x_end]
        dst_valid = res_image[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
        mask_valid = mask[src_y_start:src_y_end, src_x_start:src_x_end]
        
        blended = src_valid * mask_valid + dst_valid * (1.0 - mask_valid)
        res_image[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = blended.astype(np.uint8)

    return res_image

if __name__ == "__main__":
    from utils.landmarks import FaceLandmarks
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, "male_face_test.png")
    
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        fl = FaceLandmarks()
        lm = fl.get_landmarks(img)
        if lm:
            res = process_eyes(img, lm)
            cv2.imwrite(os.path.join(base_dir, "result_eyes.jpg"), res)
            print("Eyes done.")
