import cv2
import numpy as np

def warp_region(image, src_points, dst_points, size):
    """
    Déforme une région d'image définie par src_points vers dst_points.
    Utilisé pour affiner le nez.
    """
    # Estimation de la matrice de transformation (Affine ou Homography)
    # Pour un warping fluide local, on utilise souvent Thin Plate Spline (TPS) 
    # mais openCV a des méthodes plus simples comme getAffineTransform pour 3 points.
    # Ici, nous allons utiliser une approche plus simple : Resize de ROI pour l'approche "Cut & Paste" simple
    pass

def get_roi_from_landmarks(image, landmarks, indices, padding=10):
    """
    Extrait une Region of Interest (ROI) rectangulaire autour des landmarks donnés.
    """
    points = np.array([landmarks[i] for i in indices])
    x, y, w, h = cv2.boundingRect(points)
    
    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    roi = image[y:y+h, x:x+w]
    return roi, (x, y, w, h)

def blend_roi(image, roi, x, y):
    """
    Réinsère la ROI dans l'image (copie simple, le blending se fera avec seamlessClone dans le module si besoin)
    """
    h, w = roi.shape[:2]
    image[y:y+h, x:x+w] = roi
    return image

def apply_affine_transform(src, src_tri, dst_tri, size):
    """
    Applique une transformation affine à un triangle d'image.
    """
    try:
        warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
        dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return dst
    except Exception:
        return np.zeros((size[1], size[0], 3), dtype=src.dtype)

def morph_triangle(img1, img2, t1, t2):
    """
    Déforme un triangle t1 de img1 vers t2 dans img2.
    """
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
