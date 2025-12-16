import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config.landmark_indices import JAW_LANDMARKS

def apply_affine_transform(src, src_tri, dst_tri, size):
    try:
        warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
        dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return dst
    except Exception:
        return np.zeros((size[1], size[0], 3), dtype=src.dtype)

def morph_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r2[2], r2[3])

    if size[0] <= 0 or size[1] <= 0: return

    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
    img2_rect = img2_rect * mask

    try:
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2_rect
    except:
        pass

def process_jaw(image, landmarks):
    """
    Affine la mâchoire avec une courbe trÃ¨s douce.
    Réduction subtile (5%) et interpolation pour un contour arrondi.
    """
    
    h, w = image.shape[:2]
    img_morphed = image.copy()
    
    nose_tip = landmarks[1]
    
    # On utilise maintenant plus de points pour lisser la courbe, 
    # mais avec un facteur de déplacement variable (plus fort au milieu, faible aux bouts)
    
    # Points complets de la machoire (sans les oreilles)
    # Right Side: 234(Fixe) ... 132(Angle) ... 152(Menton)
    # Left Side: 152(Menton) ... 361(Angle) ... 454(Fixe)
    
    # Indices intermédiaires pour courbe lisse
    jaw_right_indices = [93, 132, 58, 172, 136, 150, 149, 176, 148] 
    chin_index = 152
    jaw_left_indices = [377, 400, 378, 379, 365, 397, 288, 361, 323]
    
    moving_indices = jaw_right_indices + [chin_index] + jaw_left_indices
    
    # Points Fixes
    fixed_indices = [234, 454, 1, 61, 291, 10, 127, 356] # Ajout tempes (127, 356) pour bloquer le haut
    
    fixed_manual = [
        [0, 0], [w/2, 0], [w-1, 0],
        [0, h/2], [w-1, h/2],
        [0, h-1], [w/2, h-1], [w-1, h-1]
    ]
    
    all_src = []
    all_dst = []
    
    # FACTEUR SUBTIL & VARIABLE
    base_thinning = 0.05 # 5% max
    
    for idx in moving_indices:
        pt = list(landmarks[idx])
        
        # Calcul vecteur vers le centre
        center_x = nose_tip[0]
        vec_x = center_x - pt[0]
        
        # Modulateur pour courbe douce:
        # L'effet est max aux angles de la machoire (132, 361) et diminue vers le menton et les oreilles
        # Mais pour féminiser, on veut supprimer les angles carrés.
        
        # IsAngle ? (132 ou 361) -> Max effect
        # IsChin ? (152) -> Petit effect (juste narrower)
        
        effect = base_thinning
        
        # Si c'est proche des oreilles (début/fin listes), on réduit l'effet pour lisser la transition
        # Si c'est proche de l'angle, on garde l'effet pour "gommer" l'os carré
        
        # Pour faire simple : on applique 5% partout sur la machoire basse, ce qui va arrondir globalement
        
        new_x = pt[0] + vec_x * effect
        
        all_src.append(pt)
        
        # Pas de lift menton, juste amincissement largeur
        all_dst.append([new_x, pt[1]])

    for idx in fixed_indices:
        pt = list(landmarks[idx])
        all_src.append(pt)
        all_dst.append(pt)

    # Ajout points interpolation oreilles<->machoire
    # On en ajoute plusieurs pour forcer la courbe
    era_r = landmarks[234]
    angle_r = landmarks[93] # Premier point mouvant
    # 2 points intermédiaires
    p1 = [(2*era_r[0] + angle_r[0])//3, (2*era_r[1] + angle_r[1])//3]
    p2 = [(era_r[0] + 2*angle_r[0])//3, (era_r[1] + 2*angle_r[1])//3]
    all_src.extend([p1, p2])
    all_dst.extend([p1, p2]) # Ils restent fixes ! Cela ancre la peau

    era_l = landmarks[454]
    angle_l = landmarks[323]
    p3 = [(2*era_l[0] + angle_l[0])//3, (2*era_l[1] + angle_l[1])//3]
    p4 = [(era_l[0] + 2*angle_l[0])//3, (era_l[1] + 2*angle_l[1])//3]
    all_src.extend([p3, p4])
    all_dst.extend([p3, p4])

    for pt in fixed_manual:
        all_src.append(pt)
        all_dst.append(pt)

    all_src = np.array(all_src, np.float32)
    all_dst = np.array(all_dst, np.float32)

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
