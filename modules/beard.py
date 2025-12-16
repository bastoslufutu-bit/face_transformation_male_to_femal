import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config.landmark_indices import FACE_OVAL, LIPS_LANDMARKS, JAW_LANDMARKS, NOSE_LANDMARKS, CHEEKS_LANDMARKS

def get_skin_color(image, landmarks):
    """
    Détecte la couleur moyenne de la peau (Front + Joues)
    """
    patches = []
    
    # Front
    forehead_idx = 10
    pt = landmarks[forehead_idx]
    patch = image[pt[1]-10:pt[1]+10, pt[0]-10:pt[0]+10]
    if patch.size > 0: patches.append(patch)
    
    # Joues
    cheek_l = landmarks[425] 
    patch = image[cheek_l[1]-15:cheek_l[1]+15, cheek_l[0]-15:cheek_l[0]+15]
    if patch.size > 0: patches.append(patch)
    
    cheek_r = landmarks[205]
    patch = image[cheek_r[1]-15:cheek_r[1]+15, cheek_r[0]-15:cheek_r[0]+15]
    if patch.size > 0: patches.append(patch)
    
    if not patches:
        return (200, 180, 150)
        
    means = []
    for p in patches:
        means.append(cv2.mean(p)[:3])
    
    global_mean = np.mean(means, axis=0)
    return global_mean

def process_beard(image, landmarks):
    """
    Suppression de Barbe/Moustache avec protection RENFORCÉE du nez.
    """
    
    h, w = image.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)
    
    # --- 1. ZONES ---
    
    # A. Moustache - ABAISSÉE
    # Au lieu de prendre le bas du nez (landmarks 102, 2, 331), on les descend de quelques pixels
    # pour être sûr de ne pas mordre sur le nez.
    
    nose_bottom_indices = [102, 2, 331]
    nose_bottom_pts = []
    for idx in nose_bottom_indices:
        pt = list(landmarks[idx])
        pt[1] += 5 # Décalage vers le bas de 5 pixels
        nose_bottom_pts.append(tuple(pt))
        
    lip_top = [landmarks[i] for i in [61, 40, 37, 0, 267, 270, 291]]
    
    moustache_pts = np.array(nose_bottom_pts + lip_top[::-1], np.int32)
    cv2.fillConvexPoly(full_mask, cv2.convexHull(moustache_pts), 255)
    
    # B. Barbe (Menton + Machoire)
    jaw_pts = [landmarks[i] for i in JAW_LANDMARKS['contour']]
    lip_bottom = [landmarks[i] for i in LIPS_LANDMARKS['lower']][::-1]
    
    jaw_poly = np.array(jaw_pts, np.int32)
    beard_poly = np.concatenate((jaw_poly, np.array(lip_bottom, np.int32)))
    cv2.fillPoly(full_mask, [cv2.convexHull(beard_poly)], 255)
    
    # C. Favoris
    sideburn_l = [landmarks[i] for i in [454, 323, 361, 365]]
    sideburn_r = [landmarks[i] for i in [234, 93, 132, 136]]
    
    cv2.fillConvexPoly(full_mask, cv2.convexHull(np.array(sideburn_l, np.int32)), 255)
    cv2.fillConvexPoly(full_mask, cv2.convexHull(np.array(sideburn_r, np.int32)), 255)
    
    # --- 2. EXCLUSION (NEZ + LÈVRES) ---
    
    mask_exclude = np.zeros((h, w), dtype=np.uint8)
    
    # Lèvres
    lips_all = [landmarks[i] for i in LIPS_LANDMARKS['outer']]
    cv2.fillPoly(mask_exclude, [np.array(lips_all, np.int32)], 255)
    
    # Nez - PROTECTION ÉLARGIE
    # On prend une zone large autour du nez
    nose_tip = [landmarks[i] for i in [1, 2, 98, 327, 4, 102, 218, 331, 48, 49, 279]] # Ajout 49, 279 pour ailes larges
    cv2.fillConvexPoly(mask_exclude, cv2.convexHull(np.array(nose_tip, np.int32)), 255)
    
    # DILATATION de l'exclusion
    # On dilate fort pour être sûr de repousser le traitement loin du nez et des lèvres
    # Kernel ellipse pour être doux
    kernel_exclude = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_exclude = cv2.dilate(mask_exclude, kernel_exclude, iterations=1)
    
    # Application exclusion
    full_mask = cv2.subtract(full_mask, mask_exclude)
    
    # Nettoyage
    kernel_clean = np.ones((5,5), np.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel_clean)
    
    # --- 3. TRAITEMENT ---
    
    skin_color = get_skin_color(image, landmarks)
    
    y_indices, x_indices = np.where(full_mask > 0)
    if len(y_indices) == 0: return image
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    pad = 20
    x_min, x_max = max(0, x_min-pad), min(w, x_max+pad)
    y_min, y_max = max(0, y_min-pad), min(h, y_max+pad)
    
    roi = image[y_min:y_max, x_min:x_max]
    mask_roi = full_mask[y_min:y_max, x_min:x_max]
    
    # Smoothing
    processed_roi = roi.copy()
    for _ in range(4):
        processed_roi = cv2.bilateralFilter(processed_roi, 9, 80, 80)
        
    # Coloring
    color_layer = np.zeros_like(processed_roi)
    color_layer[:] = skin_color
    
    color_mix = cv2.addWeighted(processed_roi, 0.55, color_layer, 0.45, 0)
    
    # --- 4. BLENDING ---
    mask_roi_blur = cv2.GaussianBlur(mask_roi, (31, 31), 10) # Blur un peu réduit pour pas baver sur le nez
    mask_roi_norm = cv2.merge([mask_roi_blur / 255.0] * 3)
    
    result_roi = (color_mix.astype(float) * mask_roi_norm + 
                  roi.astype(float) * (1.0 - mask_roi_norm)).astype(np.uint8)
    
    image[y_min:y_max, x_min:x_max] = result_roi
    
    return image
