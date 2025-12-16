# Face Transformation: Male to Female 

## 📌 Description

**Face Transformation: Male to Female** est un projet de **transformation faciale intelligente** basé sur la vision par ordinateur.
Il permet de transformer **un visage masculin en visage féminin**, **partie par partie**, en utilisant **exclusivement les landmarks faciaux de MediaPipe**.

Chaque composant du visage (peau, nez, lèvres, yeux, cheveux, etc.) est **indépendant**, ce qui garantit un contrôle précis, modulaire et progressif de la transformation.

Ce projet est conçu à des fins **éducatives, expérimentales et de recherche** en computer vision.

---

##  Objectifs du projet

* Comprendre et exploiter les **landmarks faciaux MediaPipe**
* Appliquer des **transformations morphologiques réalistes**
* Concevoir une architecture **modulaire** (chaque partie du visage est indépendante)
* Obtenir un rendu **progressif, naturel et contrôlé**
* Fournir une base solide pour des projets de **face editing / face morphing**

---

## Technologies utilisées

* **Python 3.8+**
* **MediaPipe** (Face Mesh)
* **OpenCV**
* **NumPy**
* **Tkinter** (interface graphique – si activée)
* **Pillow (PIL)**

---

## 🗂️ Architecture du projet

```text
face_transformation_male_to_femal/
│
├── main.py                # Point d’entrée du projet
├── config/
│   └── landmark_indices.py
│
├── modules/
│   ├── skin.py            # Transformation de la peau
│   ├── nose.py            # Transformation du nez
│   ├── lips.py            # Transformation des lèvres
│   ├── eyes.py            # Transformation des yeux
│   └── hair.py            # Gestion des cheveux / perruques
│
├── utils/
│   ├── geometry.py        # Calculs géométriques
│   └── helpers.py
│
├── assets/                # Images de test
├── outputs/               # Résultats générés
├── .gitignore
└── README.md
```

---

## ⚙️ Fonctionnalités principales

* ✔️ Détection faciale avec **MediaPipe Face Mesh**
* ✔️ Extraction précise des **landmarks**
* ✔️ Transformation **indépendante** de chaque partie du visage
* ✔️ Ajustement progressif (intensité, largeur, hauteur, finesse)
* ✔️ Compatible image & webcam
* ✔️ Architecture extensible

---

## 🚀 Installation

### 1️⃣ Cloner le dépôt

```bash
git clone https://github.com/bastoslufutu-bit/face_transformation_male_to_femal.git
cd face_transformation_male_to_femal
```

### 2️⃣ Installer les dépendances

```bash
pip install opencv-python mediapipe numpy pillow
```

---

## ▶️ Utilisation

```bash
python main.py
```

* Charge une image ou active la webcam
* Sélectionne la partie du visage à transformer
* Ajuste les paramètres (intensité, forme, finesse)
* Visualise le rendu en temps réel ou sauvegarde le résultat

---

## ⚠️ Limitations

* Le projet ne vise **pas l’usurpation d’identité**
* Les résultats dépendent fortement de la qualité de l’image
* Le réalisme final dépend des ajustements manuels

---

## 📚 Cas d’utilisation

* Recherche en **Computer Vision**
* Études sur les **landmarks faciaux**
* Projets éducatifs
* Prototypage Face Editing / Gender Morphing
* Applications artistiques

---

## 🛡️ Éthique & Responsabilité

Ce projet est destiné à un usage **éthique, pédagogique et expérimental**.
Toute utilisation abusive ou contraire à la vie privée est **fortement déconandée**.

---

## 👤 Auteur

**Bastos Lufutu**
GitHub : [https://github.com/bastoslufutu-bit](https://github.com/bastoslufutu-bit)

---

## 📄 Licence

Ce projet est sous licence **MIT** – libre d’utilisation à des fins éducatives et de recherche.
