import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config.landmark_indices import JAW_LANDMARKS
from utils.geometry import morph_triangle

def process_chin(image, landmarks):
    """
    Affine le menton (Chin) pour le rendre plus pointu et féminin.
    - Remonte légère la pointe (Lift)
    - Affine la largeur de la pointe (Narrowing)
    """
    
    h, w = image.shape[:2]
    img_morphed = image.copy()
    
    nose_tip = landmarks[1]
    
    # Points clés du menton
    # 152 : Pointe menton
    # 377, 148 : Points adjacents sur la ligne machoire
    # 378, 176 : Points un peu plus externes
    # 411, 186 : Points sur le menton même (pas ligne machoire) ? 
    # Mediapipe Chin grid : 175 (center chin), 152 (bottom)
    
    chin_tip = 152
    
    # On va bouger la pointe 152 et ses voisins immédiats
    moving_indices = [152, 148, 377, 176, 400]
    
    # Points fixes : 
    # Mâchoire plus loin (pour isoler le mouvement du menton)
    # Lèvre inférieure (pour ne pas déformer la bouche)
    fixed_indices = [132, 361, 17, 0, 164, 18] # 132/361=Angles Machoire, 17=Lèvre Inf, 0/164/18=Axe bouche
    # On ajoute aussi le nez et yeux pour contexte triangulation
    fixed_indices += [1] 
    
    fixed_manual = [
        [0, 0], [w/2, 0], [w-1, 0],
        [0, h/2], [w-1, h/2],
        [0, h-1], [w/2, h-1], [w-1, h-1]
    ]
    
    all_src = []
    all_dst = []
    
    # 1. Transformation MENTON
    # On remonte la pointe (Lift) et on la serre (Narrow)
    
    lift_factor = 0.03 # 3% de la distance nez-menton en moins (remonte)
    narrow_factor = 0.05 # 5% plus étroit autour de l'axe central
    
    center_x = nose_tip[0] # Axe symétrie
    nose_y = nose_tip[1]
    
    # Distance de référence
    chin_y = landmarks[152][1]
    ref_dist = chin_y - nose_y
    
    for idx in moving_indices:
        pt = list(landmarks[idx])
        
        # Lift vertical
        new_y = pt[1] - (ref_dist * lift_factor)
        
        # Narrow horizontal
        vec_x = center_x - pt[0]
        # On rapproche du centre
        new_x = pt[0] + vec_x * narrow_factor
        
        all_src.append(pt)
        all_dst.append([new_x, new_y])
        
    # Points fixes
    for idx in fixed_indices:
        pt = list(landmarks[idx])
        all_src.append(pt)
        all_dst.append(pt)
        
    for pt in fixed_manual:
        all_src.append(pt)
        all_dst.append(pt)
        
    all_src = np.array(all_src, np.float32)
    all_dst = np.array(all_dst, np.float32)
    
    # Warping
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    for p in all_src:
        subdiv.insert((p[0], p[1]))
    triangle_list = subdiv.getTriangleList()

    for t in triangle_list:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        triangle_indices = []
        valid_tri = True
        for pt in pts:
            # Recherche index
            dists = np.sqrt(np.sum((all_src - pt)**2, axis=1))
            min_idx = np.argmin(dists)
            if dists[min_idx] < 2.0:
                triangle_indices.append(min_idx)
            else:
                valid_tri = False
                break
        
        if valid_tri:
            t_src = all_src[triangle_indices]
            t_dst = all_dst[triangle_indices]
            morph_triangle(image, img_morphed, t_src, t_dst)
            
    return img_morphed
