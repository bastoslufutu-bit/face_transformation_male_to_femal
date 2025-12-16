import mediapipe as mp
import cv2
import numpy as np

class FaceLandmarks:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def get_landmarks(self, image):
        """
        Retourne une liste de tuples (x, y) pour les 478 landmarks.
        """
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks_list = []
            for lm in face_landmarks.landmark:
                x = int(lm.x * width)
                y = int(lm.y * height)
                landmarks_list.append((x, y))
            return landmarks_list
        return None
