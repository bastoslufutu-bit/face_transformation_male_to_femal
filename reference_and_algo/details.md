# Documentation Technique : Algorithmes et Références

Ce document détaille le fonctionnement interne des modules de transformation pour le projet Gender Transformation (Male to Female).

## 1. Module Cheveux (`modules/hair.py`)

### Référence
*   **Source** : `hairstyle.jpg` (Image locale servant de modèle de coiffure).
*   **Technologie** : OpenCV (Traitement d'image), Masques HSV.

### Algorithme
L'objectif est de remplacer la coiffure d'origine par celle de l'image de référence tout en conservant le visage de l'utilisateur.

#### Étape 1 : Extraction du Visage (Début)
L'algorithme commence par détecter les points de repère du visage (Face Landmarks) pour définir l'ovale du visage.
*   Il crée un masque binaire précis de la zone du visage.
*   Il extrait cette zone de l'image d'origine pour la réutiliser plus tard.

#### Étape 2 : Préparation de la Coiffure
*   L'image `hairstyle.jpg` est chargée.
*   Elle est redimensionnée dynamiquement pour correspondre à la largeur du visage détecté (environ 2.5x la largeur du visage).
*   **Détourage** : Un masque est généré en analysant la saturation des couleurs (HSV). Les zones colorées (cheveux) sont conservées, les fonds neutres/gris sont rendus transparents.

#### Étape 3 : Nettoyage de l'Arrière-Plan
Avant de coller la nouvelle coiffure, l'algorithme tente d'effacer les cheveux originaux qui pourraient dépasser.
*   Il détecte la couleur de fond en haut de l'image.
*   Il "peint" par-dessus la zone haute du crâne avec cette couleur de fond, tout en protégeant soigneusement le centre du visage et le cou pour ne pas effacer de traits essentiels.

#### Étape 4 : Composition (Fin)
1.  La nouvelle coiffure est superposée sur l'image (centrée par rapport au visage).
2.  Pour assurer un réalisme parfait, le visage extrait à l'Étape 1 est recollé **par-dessus** la coiffure. Cela garantit que la nouvelle coiffure passe "derrière" les oreilles ou le contour du visage si nécessaire, et évite qu'elle ne recouvre les yeux ou la bouche.

---

## 2. Module Peau (`modules/skin.py`)

### Référence
*   **Technique** : Lissage par Filtre Bilatéral (Bilateral Filtering).
*   **Inspiration** : Techniques de "Beauty Retouching" utilisées dans Photoshop.

### Algorithme
L'objectif est d'unifier le teint et de supprimer les imperfections/barbe naissante sans flouter les détails importants (yeux, bouche).

#### Étape 1 : Création des Masques (Début)
*   **Masque Global** : Tout l'ovale du visage est sélectionné grâce aux landmarks.
*   **Masques d'Exclusion** : Des zones de protection sont créées pour les Yeux, la Bouche et les Sourcils. On ne veut absolument pas flouter ces zones car cela rendrait l'image fausse ou floue.

#### Étape 2 : Lissage Intelligent
*   L'algorithme applique un **Filtre Bilatéral** sur l'image entière.
    *   *Particularité du Filtre Bilatéral* : Contrairement à un flou classique qui mélange tout, ce filtre lisse les zones où les couleurs sont proches (la peau) mais **préserve les bords nets** (les arêtes du nez, le contour du machoire).
    *   Paramètres utilisés : `sigmaColor=80`, `sigmaSpace=80` (Fort lissage pour effet féminisation).

#### Étape 3 : Fusion (Fin)
*   L'image finale est une combinaison :
    *   Les zones de peau proviennent de l'image lissée.
    *   Les yeux, la bouche et le fond proviennent de l'image originale (piksels intacts).
    *   Les bords des masques sont adoucis (Gaussian Blur sur le masque) pour une transition invisible.

---

## 3. Module Nez (`modules/nose.py`)

### Référence
*   **Technique** : "Seamless Cloning" (Poisson Blending) et Redimensionnement local.
*   **Esthétique** : Réduction de largeur (~15%) pour un nez plus fin.

### Algorithme
L'objectif est d'affiner le nez sans créer de distorsions visibles dans la texture de la peau environnante.

#### Étape 1 : Isolation (Début)
*   La zone rectangulaire contenant le nez est découpée dans l'image originale.

#### Étape 2 : Transformation Géométrique
*   Cette imagette du nez est redimensionnée horizontalement (axe X) à **85%** de sa largeur initiale.
*   Cela "compresse" le nez pour le rendre plus étroit.

#### Étape 3 : Réintégration (Fin)
*   Le défi est de remettre ce nez plus petit sans laisser de trous sur les côtés.
*   L'algorithme utilise la fonction `cv2.seamlessClone` d'OpenCV.
*   Cet algorithme mathématique (basé sur les gradients) ajuste automatiquement la luminosité et la couleur des bords du nouveau nez pour qu'il se fonde parfaitement avec la peau environnante, remplissant "magiquement" les espaces vides créés par le rétrécissement.

---

## 4. Module Menton (`modules/chin.py`)

### Référence
*   **Technique** : Warping par Triangulation de Delaunay.
*   **Esthétique** : "V-Shape" (Forme en V), menton plus pointu et moins carré.

### Algorithme
Contrairement au nez (qui est une image collée), le menton est déformé "élastiquement" (Warping).

#### Étape 1 : Définition des Points (Début)
*   L'algorithme identifie des points clés sur le menton (pointe) et la mâchoire.
*   Il définit une cible pour chaque point :
    *   **Lift** : La pointe du menton est remontée vers le haut (3%).
    *   **Narrow** : Les côtés du menton sont rapprochés vers le centre (5%).

#### Étape 2 : Triangulation
*   L'image est divisée en une multitude de petits triangles reliant ces points clés (Maillage).

#### Étape 3 : Déformation (Warping)
*   Chaque triangle de l'image source est déformé géométriquement pour correspondre à la nouvelle position des points.
*   La texture de la peau à l'intérieur de chaque triangle est étirée ou compressée de manière fluide.

#### Étape 4 : Reconstruction (Fin)
*   L'image est reconstruite triangle par triangle. Le résultat est un menton structurellement modifié, sans coupure ni collage, comme si la structure osseuse avait changé.
