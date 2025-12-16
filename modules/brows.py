import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config.landmark_indices import BROWS_LANDMARKS
from utils.geometry import get_roi_from_landmarks

def process_single_brow(image, landmarks, indices, is_left=True):
    """
    Traite un sourcil :
    1. Efface l'ancien (Inpainting Fort).
    2. Redessine un NOUVEAU sourcil synthétique :
       - Fin
       - Arqué
       - Bien tracé / Maquillé
    """
    
    # 1. Masquage étendu pour effacer complètement l'ancien sourcil
    mask_brow = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array([landmarks[i] for i in indices], np.int32)
    
    # Zone large pour effacer tous les poils
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask_brow, hull, 255)
    kernel = np.ones((7,7), np.uint8)
    mask_brow = cv2.dilate(mask_brow, kernel, iterations=3) # Dilatation forte
    
    # Inpainting pour effacer
    image_no_brow = cv2.inpaint(image, mask_brow, 15, cv2.INPAINT_TELEA)
    
    # 2. Dessiner le nouveau sourcil
    # On utilise les points originaux comme référence de position, mais on modifie la courbe
    
    # Trier les points de gauche à droite
    sorted_pts = points[np.argsort(points[:, 0])]
    
    # Définir 3 points clés pour une courbe de Bézier quadratique (Début, Sommet de l'arche, Fin)
    start_pt = sorted_pts[0]
    end_pt = sorted_pts[-1]
    
    # Le sommet (arche) doit être plus haut et au tiers extérieur
    # Calcul du milieu
    mid_x = (start_pt[0] + end_pt[0]) // 2
    
    # Arche : décalage vers le haut
    # Pour sourcil gauche (droite image) : Sommet vers la fin
    # Pour sourcil droit (gauche image) : Sommet vers le début
    
    # Largeur totale
    width = end_pt[0] - start_pt[0]
    
    peak_x = start_pt[0] + width // 3 if not is_left else start_pt[0] + (2 * width) // 3
    
    # Hauteur : On remonte le sommet par rapport à la moyenne Y
    avg_y = (start_pt[1] + end_pt[1]) // 2
    lift_amount = int(width * 0.15) # L'arche monte de 15% de la largeur
    peak_y = avg_y - lift_amount
    
    # Ajustement start/end pour qu'ils soient un peu plus hauts que l'original aussi (effet lifté)
    start_pt[1] -= 2
    end_pt[1] -= 2
    
    # Points de contrôle pour la courbe
    # On utilise polylines avec curve fit ou simplement cv2.line pour un style "tracé"
    # Mieux : un polygone rempli pour l'épaisseur variable
    
    # Style "Femme" : Plus épais au nez, très fin à la queue
    
    # On va générer des points interpolés
    brow_curve = []
    steps = 20
    for t in np.linspace(0, 1, steps):
        # Interpolation quadratique de Bézier
        # P = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
        # P0 = Start, P1 = Peak control point, P2 = End
        
        # Le point de contrôle P1 doit être plus haut que le peak pour que la courbe passe par le peak ?
        # Approx : P1 est au dessus du peak.
        p1_x = peak_x
        p1_y = peak_y - lift_amount // 2
        
        x = (1-t)**2 * start_pt[0] + 2*(1-t)*t * p1_x + t**2 * end_pt[0]
        y = (1-t)**2 * start_pt[1] + 2*(1-t)*t * p1_y + t**2 * end_pt[1]
        brow_curve.append((int(x), int(y)))
    
    # Dessin du sourcil sur un calque transparent
    overlay = image_no_brow.copy()
    
    # Tracé progressif de l'épaisseur
    for i in range(len(brow_curve) - 1):
        pt1 = brow_curve[i]
        pt2 = brow_curve[i+1]
        
        # Épaisseur diminue de gauche à droite (si oeil droit) ou augmente (si oeil gauche ?)
        # Non, épais au centre du visage (nez), fin vers l'extérieur
        
        # Distance relative au nez
        # Si is_left (oeil gauche du sujet = droite image) : Nez est à gauche du sourcil (x min)
        # Si !is_left (oeil droit du sujet = gauche image) : Nez est à droite du sourcil (x max)
        
        # En fait sorted_pts va de petit x à grand x (Gauche image -> Droite image)
        
        # Oeil Droit Sujet (Gauche Image) : Nez est à droite (fin de liste sorted) -> Epais à la fin
        # Oeil Gauche Sujet (Droite Image) : Nez est à gauche (début de liste sorted) -> Epais au début
        
        progress = i / steps
        # Agrandissement de l'épaisseur (était 4)
        base_thickness = 7
        
        if is_left: # Droite image. Nez à gauche (début). Epais -> Fin
            thickness = max(2, int(base_thickness * (1 - progress)))
        else: # Gauche image. Nez à droite (fin). Fin -> Epais
             thickness = max(2, int(base_thickness * progress))
             
        cv2.line(overlay, pt1, pt2, (20, 15, 15), thickness, cv2.LINE_AA)
        
    # Fusion (Opacity 0.7 pour laisser un peu de "naturel" si besoin, ou 1.0 pour tracé net)
    cv2.addWeighted(overlay, 0.85, image_no_brow, 0.15, 0, image_no_brow)
    
    return image_no_brow

def process_brows(image, landmarks):
    """
    Redessine les sourcils style féminin
    """
    image = process_single_brow(image, landmarks, BROWS_LANDMARKS['left'], is_left=True)
    image = process_single_brow(image, landmarks, BROWS_LANDMARKS['right'], is_left=False)
    return image
