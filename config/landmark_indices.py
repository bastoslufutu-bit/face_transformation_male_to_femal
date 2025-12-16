# Indices Mediapipe Face Mesh pour chaque partie du visage

# LIPS (Levres)
LIPS_LANDMARKS = {
    "upper": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    "lower": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "outer": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375]
}

# EYES (Yeux)
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYES_LANDMARKS = {
    "left": LEFT_EYE,
    "right": RIGHT_EYE
}

# BROWS (Sourcils)
LEFT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
BROWS_LANDMARKS = {
    "left": LEFT_EYEBROW,
    "right": RIGHT_EYEBROW
}

# NOSE (Nez)
# Bridge (arete) and Tip (pointe)
NOSE_LANDMARKS = {
    "bridge": [1, 2, 98, 327],
    "tip": [4],
    "nostrils": [102, 218, 331, 48] # Ailes du nez
}

# JAW & CHIN (Machoire et Menton)
JAW_LANDMARKS = {
    "contour": [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
}

# CHEEKS (Joues)
# Approximation des zones des pommettes
CHEEKS_LANDMARKS = {
    "left": [425, 266, 329, 348], # Zone gauche
    "right": [205, 36, 100, 119]  # Zone droite
}

# FACE OVAL (Pour la peau)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
