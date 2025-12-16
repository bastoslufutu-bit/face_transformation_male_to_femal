import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config.landmark_indices import FACE_OVAL

def process_hair(image, landmarks):
    """
    NOUVELLE APPROCHE :
    1. Extraire uniquement le visage (landmarks) de l'image d'entrée
    2. Coller ce visage sur hairstyle.jpg comme fond
    """
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hairstyle_path = os.path.join(base_dir, "hairstyle.jpg")
    
    if not os.path.exists(hairstyle_path):
        return image
    
    hairstyle = cv2.imread(hairstyle_path)
    if hairstyle is None:
        return image

    h_hair, w_hair = hairstyle.shape[:2]
    h_img, w_img = image.shape[:2]
    
    # --- 1. CRÉER MASQUE DU VISAGE (FACE_OVAL) ---
    face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    face_points = np.array([landmarks[i] for i in FACE_OVAL], np.int32)
    cv2.fillPoly(face_mask, [face_points], 255)
    
    # Dilater légèrement pour inclure les bords
    kernel = np.ones((5, 5), np.uint8)
    face_mask = cv2.dilate(face_mask, kernel, iterations=2)
    
    # Flou pour adoucir les bords
    face_mask = cv2.GaussianBlur(face_mask, (15, 15), 0)
    
    # --- 2. EXTRAIRE LE VISAGE UNIQUEMENT ---
    # On garde uniquement la zone du visage
    face_only = cv2.bitwise_and(image, image, mask=face_mask)
    
    # --- 3. CALCULER BOUNDING BOX DU VISAGE ---
    x, y, w, h = cv2.boundingRect(face_points)
    
    # Crop le visage + masque
    face_crop = face_only[y:y+h, x:x+w]
    mask_crop = face_mask[y:y+h, x:x+w]
    
    # --- 4. REDIMENSIONNER HAIRSTYLE POUR CORRESPONDRE À LA TAILLE DU VISAGE ---
    # On veut que hairstyle soit proportionnel au visage
    # On redimensionne hairstyle pour que sa largeur soit ~2x celle du visage
    target_width = int(w * 2.5)
    aspect_ratio = h_hair / w_hair
    target_height = int(target_width * aspect_ratio)
    
    hairstyle_resized = cv2.resize(hairstyle, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    
    # --- MASQUE ULTRA-SIMPLE : SATURATION UNIQUEMENT ---
    # Fond blanc/beige/gris = Saturation FAIBLE
    # Cheveux colorés = Saturation FORTE
    
    hsv = cv2.cvtColor(hairstyle_resized, cv2.COLOR_BGR2HSV)
    saturation = hsv[:,:,1]
    
    # Seuil strict : si saturation < 60 → transparent (fond)
    _, hair_mask = cv2.threshold(saturation, 60, 255, cv2.THRESH_BINARY)
    
    # Nettoyage morphologique
    kernel = np.ones((3, 3), np.uint8)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Flou minimal
    hair_mask = cv2.GaussianBlur(hair_mask, (3, 3), 0)

    
    # --- 5. NETTOYAGE HAUT TÊTE UNIQUEMENT (INVERSE NUQUE) ---
    # "Fais exactement l'inverse" : On efface le HAUT, on garde le BAS (Nuque).
    
    # 1. Détection couleur du fond
    top_h = max(10, h_img // 20)
    bg_strip = image[0:top_h, :]
    bg_B = np.median(bg_strip[:,:,0])
    bg_G = np.median(bg_strip[:,:,1])
    bg_R = np.median(bg_strip[:,:,2])
    bg_color = [int(bg_B), int(bg_G), int(bg_R)]
    
    # 2. Masque "Zone à effacer" (HAUT TÊTE)
    clean_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    
    all_lm = np.array([landmarks[i] for i in range(468)], np.int32)
    x_f, y_f, w_f, h_f = cv2.boundingRect(all_lm)
    
    # Ellipse Tête Globale (On vise le haut)
    center_head = (x_f + w_f//2, y_f + int(h_f * 0.4)) # On centre un peu plus haut
    axes_head = (int(w_f * 0.9), int(h_f * 0.7)) # Ellipse qui couvre le haut
    cv2.ellipse(clean_mask, center_head, axes_head, 0, 0, 360, 255, -1)
    
    # Zone d'effacement rectangulaire pour les côtés hauts
    start_y_erase = int(y_f + h_f * 0.6) # Ligne des oreilles environs
    
    # 3. Protections (Ce qu'on GARDES)
    
    # Protection BAS (Nuque) - L'inverse de tout à l'heure
    # Tout ce qui est en dessous de la ligne des oreilles est protégé
    cv2.rectangle(clean_mask, (0, start_y_erase), (w_img, h_img), 0, -1)
    
    # Protection Visage (Ovale)
    cv2.fillPoly(clean_mask, [face_points], 0)
    
    # Protection Cou (Redondant avec la protection BAS mais sécurité)
    chin_pt = landmarks[152]
    neck_w = int(w_f * 0.55) 
    neck_x = chin_pt[0] - neck_w // 2
    cv2.rectangle(clean_mask, (neck_x, chin_pt[1]), (neck_x + neck_w, h_img), 0, -1)
    
    # 4. Application
    result = image.copy()
    result[clean_mask == 255] = bg_color
    
    # --- 6. POSITIONNER ET COLLER HAIRSTYLE (CHEVEUX) SUR LE FOND ORIGINAL ---
    h_hair_new, w_hair_new = hairstyle_resized.shape[:2]
    
    # Centre approximatif du visage dans l'image originale
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    
    # Position de hairstyle (centré sur le visage)
    top_left_x = face_center_x - w_hair_new // 2
    top_left_y = face_center_y - h_hair_new // 2
    
    # Gestion des bords
    y1 = max(0, top_left_y)
    y2 = min(h_img, top_left_y + h_hair_new)
    x1 = max(0, top_left_x)
    x2 = min(w_img, top_left_x + w_hair_new)
    
    # Crop correspondant dans hairstyle
    hair_y1 = y1 - top_left_y
    hair_y2 = hair_y1 + (y2 - y1)
    hair_x1 = x1 - top_left_x
    hair_x2 = hair_x1 + (x2 - x1)
    
    if hair_y2 > hair_y1 and hair_x2 > hair_x1:
        hair_slice = hairstyle_resized[hair_y1:hair_y2, hair_x1:hair_x2]
        mask_hair_slice = hair_mask[hair_y1:hair_y2, hair_x1:hair_x2]
        
        # Normaliser le masque
        alpha_hair = mask_hair_slice.astype(float) / 255.0
        alpha_hair_3c = cv2.merge([alpha_hair] * 3)
        
        # Blend cheveux sur fond original
        bg_slice = result[y1:y2, x1:x2]
        blended_hair = (hair_slice.astype(float) * alpha_hair_3c + 
                        bg_slice.astype(float) * (1.0 - alpha_hair_3c)).astype(np.uint8)
        
        result[y1:y2, x1:x2] = blended_hair
    
    
    # --- 7. COLLER LE VISAGE PAR-DESSUS (MASQUE RÉDUIT) ---
    # IMPORTANT : On réduit le masque pour ne recoller QUE la peau centrale
    # Sans les bords qui peuvent contenir les cheveux originaux
    
    # Érosion FORTE pour réduire le masque au strict minimum (centre du visage)
    kernel_reduce = np.ones((25, 25), np.uint8)
    face_mask_reduced = cv2.erode(face_mask, kernel_reduce, iterations=2)
    
    # Flou pour adoucir les transitions
    face_mask_reduced = cv2.GaussianBlur(face_mask_reduced, (21, 21), 0)
    
    alpha_face = face_mask_reduced.astype(float) / 255.0
    alpha_face_3c = cv2.merge([alpha_face] * 3)
    
    result = (image.astype(float) * alpha_face_3c + 
              result.astype(float) * (1.0 - alpha_face_3c)).astype(np.uint8)
    
    return result

if __name__ == "__main__":
    pass
