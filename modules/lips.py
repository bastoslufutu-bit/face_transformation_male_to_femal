import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config.landmark_indices import LIPS_LANDMARKS
from utils.geometry import get_roi_from_landmarks


def process_lips(image, landmarks):
    """
    Version VOLUME UNIQUEMENT avec MASQUE PRÉCIS :
    - Pas de coloration (naturel)
    - Augmentation de volume (+15%)
    - Intégration via masque précis des landmarks (pas de débordement)
    """

    indices = LIPS_LANDMARKS["outer"]
    roi, (x, y, w, h) = get_roi_from_landmarks(image, landmarks, indices, padding=2)

    if roi.size == 0 or w < 5 or h < 5:
        return image

    # 1. Créer le masque précis des lèvres à partir des landmarks
    lip_points = []
    for idx in indices:
        pt = landmarks[idx]
        lip_points.append((pt[0] - x, pt[1] - y))
    
    lip_points_array = np.array(lip_points, np.int32)
    
    lip_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lip_mask, [lip_points_array], 255)
    
    # Flou léger pour bords doux
    lip_mask_blur = cv2.GaussianBlur(lip_mask, (5, 5), 0)

    # 2. COLORATION (Rose)
    # Couleur rose/rouge demandée
    lipstick_color = np.array([165, 50, 210], dtype=np.uint8)
    lipstick_overlay = np.zeros_like(roi)
    lipstick_overlay[:] = lipstick_color
    
    # Mélange 30% couleur
    roi = cv2.addWeighted(roi, 0.7, lipstick_overlay, 0.3, 0)

    # 3. PLUMPING (Volume)
    
    scale_factor = 1.15
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    # Redimensionner l'image ET le masque
    plumped_img = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    plumped_mask = cv2.resize(lip_mask_blur, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 3. INTÉGRATION
    img_h, img_w = image.shape[:2]
    src_h, src_w = plumped_img.shape[:2]

    center_x, center_y = x + w // 2, y + h // 2
    tl_x = center_x - src_w // 2
    tl_y = center_y - src_h // 2
    
    # Calcul des coordonnées destination
    dst_x_start = max(0, tl_x)
    dst_y_start = max(0, tl_y)
    dst_x_end = min(img_w, tl_x + src_w)
    dst_y_end = min(img_h, tl_y + src_h)

    if dst_x_start >= dst_x_end or dst_y_start >= dst_y_end:
        return image

    # Calcul des coordonnées source correspondantes
    src_x_start = dst_x_start - tl_x
    src_y_start = dst_y_start - tl_y
    src_x_end = src_x_start + (dst_x_end - dst_x_start)
    src_y_end = src_y_start + (dst_y_end - dst_y_start)

    # Extraction des zones valides
    src_valid = plumped_img[src_y_start:src_y_end, src_x_start:src_x_end]
    dst_valid = image[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
    mask_valid = plumped_mask[src_y_start:src_y_end, src_x_start:src_x_end]

    # Normalisation du masque (0-1) et passage en 3 canaux
    mask_norm = mask_valid / 255.0
    mask_3c = cv2.merge([mask_norm] * 3)

    # Mélange alpha avec le masque précis redimensionné
    blended = src_valid.astype(float) * mask_3c + dst_valid.astype(float) * (1.0 - mask_3c)
    image[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = blended.astype(np.uint8)

    return image


if __name__ == "__main__":
    from utils.landmarks import FaceLandmarks

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, "male_face_test.png")

    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        fl = FaceLandmarks()
        lm = fl.get_landmarks(img)
        if lm:
            res = process_lips(img, lm)
            cv2.imwrite(os.path.join(base_dir, "result_lips.jpg"), res)
            print("Lips done.")
