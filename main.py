import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import sys

# Ajouter le chemin courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.landmarks import FaceLandmarks
from modules.nose import process_nose
from modules.skin import process_skin
from modules.lips import process_lips
from modules.brows import process_brows
from modules.hair import process_hair
from modules.beard import process_beard
from modules.eyes import process_eyes
from modules.jaw import process_jaw
from modules.cheeks import process_cheeks
from modules.chin import process_chin

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transformation M2F - Projet Graduation")
        self.geometry("1100x600")
        
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.landmarks = None
        self.fl = FaceLandmarks()
        
        self.create_widgets()
        
    def create_widgets(self):
        # Frame de commandes (Haut)
        control_frame = tk.Frame(self, padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Style des boutons
        btn_config = {'font': ('Arial', 10, 'bold'), 'padx': 10, 'pady': 5}
        
        btn_load = tk.Button(control_frame, text="Charger Photo", command=self.load_image, bg="#4CAF50", fg="white", **btn_config)
        btn_load.pack(side=tk.LEFT, padx=5)
        
        btn_nose = tk.Button(control_frame, text="Nez", command=lambda: self.apply_module(process_nose, "Nez"), bg="#2196F3", fg="white", **btn_config)
        btn_nose.pack(side=tk.LEFT, padx=5)

        btn_skin = tk.Button(control_frame, text="Peau", command=lambda: self.apply_module(process_skin, "Peau"), bg="#E91E63", fg="white", **btn_config)
        btn_skin.pack(side=tk.LEFT, padx=5)

        btn_lips = tk.Button(control_frame, text="Lèvres", command=lambda: self.apply_module(process_lips, "Lèvres"), bg="#9C27B0", fg="white", **btn_config)
        btn_lips.pack(side=tk.LEFT, padx=5)

        btn_brows = tk.Button(control_frame, text="Sourcils", command=lambda: self.apply_module(process_brows, "Sourcils"), bg="#795548", fg="white", **btn_config)
        btn_brows.pack(side=tk.LEFT, padx=5)

        btn_hair = tk.Button(control_frame, text="Cheveux", command=lambda: self.apply_module(process_hair, "Cheveux"), bg="#FF5722", fg="white", **btn_config)
        btn_hair.pack(side=tk.LEFT, padx=5)

        btn_beard = tk.Button(control_frame, text="Barbe", command=lambda: self.apply_module(process_beard, "Barbe"), bg="#8D6E63", fg="white", **btn_config)
        btn_beard.pack(side=tk.LEFT, padx=5)

        btn_eyes = tk.Button(control_frame, text="Yeux", command=lambda: self.apply_module(process_eyes, "Yeux"), bg="#9C27B0", fg="white", **btn_config)
        btn_eyes.pack(side=tk.LEFT, padx=5)

        btn_jaw = tk.Button(control_frame, text="Mâchoire", command=lambda: self.apply_module(process_jaw, "Mâchoire"), bg="#009688", fg="white", **btn_config)
        btn_jaw.pack(side=tk.LEFT, padx=5)

        btn_cheeks = tk.Button(control_frame, text="Joues", command=lambda: self.apply_module(process_cheeks, "Joues"), bg="#E91E63", fg="white", **btn_config)
        btn_cheeks.pack(side=tk.LEFT, padx=5)

        btn_chin = tk.Button(control_frame, text="Menton", command=lambda: self.apply_module(process_chin, "Menton"), bg="#607D8B", fg="white", **btn_config)
        btn_chin.pack(side=tk.LEFT, padx=5)

        btn_reset = tk.Button(control_frame, text="Réinitialiser", command=self.reset_image, bg="#607D8B", fg="white", **btn_config)
        btn_reset.pack(side=tk.LEFT, padx=5)
        
        btn_save = tk.Button(control_frame, text="Sauvegarder", command=self.save_image, bg="#FF9800", fg="white", **btn_config)
        btn_save.pack(side=tk.LEFT, padx=5)
        
        # Frame principale pour les images (Milieu)
        self.image_frame = tk.Frame(self, bg="#eeeeee")
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel Gauche (Original)
        self.panel_left = tk.Label(self.image_frame, text="Image Originale", bg="#cccccc")
        self.panel_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Panel Droite (Resultat)
        self.panel_right = tk.Label(self.image_frame, text="Résultat", bg="#cccccc")
        self.panel_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Erreur", "Impossible de charger l'image.")
                return
                
            self.original_image = image.copy()
            self.processed_image = image.copy() # Reset result
            
            # Detect landmarks une seule fois au chargement
            self.landmarks = self.fl.get_landmarks(self.original_image)
            if not self.landmarks:
                messagebox.showwarning("Attention", "Aucun visage détecté sur cette photo !")
            
            self.display_image(self.original_image, self.panel_left)
            self.display_image(self.processed_image, self.panel_right)
            
    def display_image(self, cv_img, panel):
        # Convert CV2 BGR to RGB
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize pour affichage (max 400x400 approx)
        base_width = 400
        w_percent = (base_width / float(img_pil.size[0]))
        if w_percent < 1: # Seulement si l'image est plus grande
            h_size = int((float(img_pil.size[1]) * float(w_percent)))
            img_pil = img_pil.resize((base_width, h_size), Image.Resampling.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(img_pil)
        panel.config(image=img_tk, text="") # Enlever le texte placeholder
        panel.image = img_tk # Garder une référence pour éviter le garbage collection

    def apply_module(self, transform_func, module_name):
        print(f"--- Début Application Module : {module_name} ---")
        if self.original_image is None:
            print("Aucune image chargée.")
            messagebox.showwarning("Attention", "Veuillez charger une image d'abord.")
            return
            
        if not self.landmarks:
            print("L'attribut self.landmarks est vide.")
            messagebox.showerror("Erreur", "Pas de landmarks détectés.")
            return

        try:
            print(f"Copie de l'image pour {module_name}...")
            img_to_process = self.processed_image.copy()
            
            print(f"Appel de la fonction {transform_func.__name__}...")
            # Application du module
            result = transform_func(img_to_process, self.landmarks)
            print(f"Retour de {transform_func.__name__}.")
            
            if result is None:
                print("ERREUR: La fonction a renvoyé None !")
                raise ValueError("La transformation a échoué (None returned)")
            
            self.processed_image = result
            print("Mise à jour de l'affichage...")
            self.display_image(self.processed_image, self.panel_right)
            print(f"Module {module_name} appliqué avec succès.")
            
        except Exception as e:
            print(f"EXCEPTION DANS MAIN.PY ({module_name}): {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Erreur", f"Erreur dans {module_name} :\n{str(e)}")

    def save_image(self):
        if self.processed_image is None:
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            messagebox.showinfo("Info", "Image sauvegardée.")
    
    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.display_image(self.processed_image, self.panel_right)


if __name__ == "__main__":
    app = Application()
    app.mainloop()
