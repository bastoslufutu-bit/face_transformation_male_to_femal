# Projet de Transformation Faciale : Homme vers Femme

Ce projet implémente une transformation de visage ("Male to Female") en utilisant des techniques de traitement d'image pure (pas de GANs/Deep Learning génératif) basées sur les landmarks faciaux détectés par **Mediapipe**.

## Objectif

Modifier les traits d'un visage masculin partie par partie pour lui donner une apparence féminine, en utilisant des déformations géométriques et des ajustements colorimétriques ciblés.

## Structure du Projet

L'architecture est modulaire, chaque partie du visage est gérée par un module dédié.

- `main.py` : Script principal qui orchestre la transformation.
- `utils/` :
    - `landmarks.py` : Gestion de la détection faciale avec Mediapipe (Face Mesh).
    - `geometry.py` : Fonctions mathématiques pour le warping (déformation) d'image.
- `modules/` :
    - `skin.py` : Lissage de la peau, réduction de la barbe.
    - `brows.py` : Affinement et rehaussement des sourcils.
    - `eyes.py` : Agrandissement des yeux.
    - `nose.py` : Affinement du nez.
    - `lips.py` : Repulpage et coloration des lèvres.
    - `jaw.py` : Affinement de la mâchoire (V-shape).
    - `cheeks.py` : Rehaussement des pommettes.

## Installation

1. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

```bash
python main.py --image path/to/image.jpg
```
