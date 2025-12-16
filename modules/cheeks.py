import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config.landmark_indices import CHEEKS_LANDMARKS

def process_cheeks(image, landmarks):
    """
    Féminisation des joues :
    1. Ajout de Blush (rosé) sur les pommettes.
    2. Ajout de Highlight (Lumière) pour accentuer le volume.
    """
    
    # Landmarks clés pour le centre des pommettes
    # Gauche : 425 (centre pommette), Droite : 205
    # On peut aussi utiliser 50 (G) et 280 (D) pour référence.
    
    idx_left = 425
    idx_right = 205
    
    centers = [landmarks[idx_left], landmarks[idx_right]]
    
    # Paramètres Blush
    blush_color_bgr = [130, 110, 200] # Rose/Mauve doux
    blush_radius_factor = 0.12 # Rayon relatif à la largeur visage ? Non, relatif distance yeux.
    # On va calculer le rayon dynamiquement
    
    # Distance entre les joues pour échelle
    dist = np.sqrt((centers[0][0]-centers[1][0])**2 + (centers[0][1]-centers[1][1])**2)
    radius = int(dist * 0.18) # Rayon de la tache de blush
    
    overlay = image.copy()
    
    for center in centers:
        # Masque circulaire flou pour le blush
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        # Flou très fort pour diffusion naturelle
        mask_blur = cv2.GaussianBlur(mask, (61, 61), 30)
        mask_norm = cv2.merge([mask_blur / 255.0] * 3)
        
        # 1. BLUSH (Couleur)
        # Créer image couleur unie
        color_img = np.zeros_like(image)
        color_img[:] = blush_color_bgr
        
        # Mélange Overlay Soft Light ou juste AddWeighted
        # AddWeighted sur la zone du masque
        # On extrait la zone dans l'image originale
        # Mais pour faire simple : on mixe l'image entière avec la couleur, selon le masque
        
        # Formule : Pixel = Original * (1 - opacity*mask) + Couleur * (opacity*mask)
        opacity = 0.25 # Subtil
        
        # On applique le mixage
        # Mais attention, un blush c'est aussi assombrir un peu ou multiplier ? 
        # Non, c'est ajouter de la teinte.
        
        blended = cv2.addWeighted(overlay, 1.0, color_img, 0.4, 0) # 0.4 densité couleur
        
        # Masquage
        overlay = (blended.astype(float) * mask_norm * opacity + 
                   overlay.astype(float) * (1.0 - mask_norm * opacity)).astype(np.uint8)
                   
        # 2. HIGHLIGHT (Volume)
        # Un point plus petit, plus haut et plus vers l'intérieur (vers le nez/yeux)
        # Pour donner l'effet bombé
        
        # Décalage highlight
        hl_radius = int(radius * 0.6)
        hl_center = (center[0], center[1] - int(radius*0.4)) 
        
        mask_hl = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask_hl, hl_center, hl_radius, 255, -1)
        mask_hl_blur = cv2.GaussianBlur(mask_hl, (41, 41), 15)
        mask_hl_norm = cv2.merge([mask_hl_blur / 255.0] * 3)
        
        # Highlight = Eclaircir (Additif)
        highlight_layer = np.full_like(image, 40) # Ajout de gris/blanc
        
        hl_opacity = 0.2
        brightened = cv2.add(overlay, highlight_layer)
        
        overlay = (brightened.astype(float) * mask_hl_norm * hl_opacity + 
                   overlay.astype(float) * (1.0 - mask_hl_norm * hl_opacity)).astype(np.uint8)

    return overlay

if __name__ == "__main__":
    from utils.landmarks import FaceLandmarks
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, "male_face_test.png")
    
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        fl = FaceLandmarks()
        lm = fl.get_landmarks(img)
        if lm:
            res = process_cheeks(img, lm)
            cv2.imwrite(os.path.join(base_dir, "result_cheeks.jpg"), res)
            print("Cheeks done.")
