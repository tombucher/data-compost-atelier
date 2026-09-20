# Atelier

Instrument de composition par saillance et dégradation. Plusieurs images en
entrée, des réglages, une image en sortie — à la taille qu'on veut.

Reprend le moteur de `blending-image2.py`, qui a produit les 230 œuvres des
six séries de décembre 2024, en levant ce qui empêchait d'en faire un outil.

## Ce que ça change

**La composition ne dépend plus de la résolution.** On explore librement en
basse définition, et un tirage retenu se relance en 4000 ou 8000 px sans rien
perdre. Les 230 œuvres existantes sont bloquées en 1024×1024, soit 8,7 cm à
300 dpi ; le même travail sort maintenant à 34 cm et au-delà.

**Le hasard reste entier.** Sans graine, chaque rendu est une composition que
personne n'a demandée. La graine ne sert qu'à revenir sur un tirage pour
l'agrandir — elle n'oblige à rien répéter.

**Rien ne s'accumule.** La graine et les réglages voyagent dans les
métadonnées de l'image, jamais dans un fichier à côté. Une image reste un
fichier et porte de quoi se refaire. Ce qui n'est pas exporté disparaît.

## Démarrer

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python demarrer.py ~/Documents/mes-images
```

Le navigateur s'ouvre sur l'atelier. Rien n'est hébergé : le serveur tourne en
local, les images ne quittent pas le disque.

| Option | Effet |
|---|---|
| `--sorties DOSSIER` | Où écrire les tirages (`sorties/` par défaut) |
| `--port N` | Port du serveur (8771 par défaut) |
| `--sans-navigateur` | Ne pas ouvrir le navigateur |

> `opencv-contrib-python`, pas `opencv-python` : `cv2.saliency` n'est pas dans
> le paquet de base. Sans lui le moteur bascule sur un repli par contraste
> local, qui ne donne pas le même rendu.

## Utiliser

À gauche, cliquer les images dans l'ordre où elles doivent se superposer. À
droite, activer les matières et régler. L'aperçu se recalcule en continu —
environ 150 ms sur trois images.

Quand une composition tient, choisir un format et **Tirer**. Le fichier part
dans `sorties/`, avec sa recette à l'intérieur.

Un tirage déjà fait réapparaît sous **Reprendre** : un clic restaure ses
images, sa graine et tous ses réglages. C'est le seul usage de la graine.

## Les matières

Les effets mordent sur les zones **peu** saillantes, sauf la saturation qui
fait l'inverse et rehausse ce qui attire l'œil. Cette asymétrie fait l'image :
le sujet reste net et se sature, le fond se décompose.

| Matière | Ce qu'elle fait |
|---|---|
| Pixellisation | Blocs de couleur moyenne |
| Bitmap | Bascule en noir et blanc, mélangée à l'original |
| Saturation | Rehausse les couleurs du sujet |
| Trame | Points de taille variable, façon similigravure |

Le grain de la pixellisation et de la trame est exprimé en fraction de la
toile, pas en pixels : il garde la même allure à toutes les tailles.

### Le débordement de saturation

Dans l'original, la saturation était multipliée dans un tableau `uint8` : au
delà de 255 les valeurs repassaient par le bas — 150 × 2 donnait 44 — et les
couleurs cassaient. Ce n'était pas voulu, mais seize œuvres de
`serie1_grece/SATURATE` en vivent.

C'est donc devenu un réglage, actif par défaut, avec le calcul écrit
explicitement pour qu'aucune mise à jour de numpy ne le change. Décoché, la
saturation plafonne normalement à 255.

## Tests

```bash
venv/bin/python -m pytest tests/ -q
```

`tests/test_fidelity.py` contient les implémentations d'origine recopiées
telles quelles et compare le moteur à elles : moins de 2 % de pixels
différents pour la trame, identique à l'octet près pour la saturation. Le
reste vérifie qu'une graine donne la même composition à toutes les tailles, et
que la recette survit à l'aller-retour dans l'image.

## La voie vidéo

Bascule **Vidéo** en haut à droite. Les mêmes images et les mêmes matières,
plus une transition : la première image se décompose par seuils de saillance
pendant que la suivante se révèle dessous, avec un tri de pixels par segments.

Le curseur **Position** parcourt la séquence. Chaque image se calcule seule,
sans rejouer ce qui précède — on s'arrête sur l'instant voulu et **Tirer cette
image** le sort à la taille choisie.

C'est ce qui remplace la capture d'écran. Les 116 images de
`export-vid_sauv/SCREENSHOT` font 2254 px alors que leurs vidéos sources en
contenaient 3016 : la capture perdait un quart de la définition. Ici l'instant
sort à 4000 ou 8000 px.

**Tirer la séquence** écrit un mp4. Le calcul dure des minutes en grand
format, il tourne en tâche de fond et l'avancement s'affiche.

### Le canal rouge n'est jamais trié

Dans le traitement par canal de l'original, chaque canal est répliqué en gris
avant d'être trié. Or trier un gris par teinte ou par saturation ne change
rien : ces valeurs y sont toutes nulles. Comme le canal rouge ne tirait
qu'entre « saturation » et « teinte », il n'était jamais trié — et le vert et
le bleu une fois sur deux.

Cette asymétrie est ce qui sépare les couleurs et fait la signature de
l'effet. Elle est conservée, comme le débordement de saturation.

## Limites connues

- Quand la cellule de trame ne divise pas la toile, la dernière cellule
  partielle peut rester sombre : son point est centré dans la cellule
  complétée, dont le centre peut tomber hors de l'image. Borné à une cellule.
- Le tri de pixels n'est vectorisé qu'à hauteur de ×1,8 : l'essentiel du
  travail était déjà dans numpy chez l'original, seule la détection des
  segments était en Python. Une séquence de 150 images en 3000 px demande
  environ cinq minutes.
- L'interface n'a été éprouvée que sur sept images sources. La colonne de
  vignettes n'a pas été essayée sur un dossier fourni.

## Pourquoi un dépôt séparé

`data-compost` est une installation : une machine qui tourne seule sur
Raspberry Pi, sous contrainte de robustesse et de matériel. L'atelier est un
instrument qu'on pilote, sous contrainte de vitesse d'essai et de qualité de
sortie. Les deux partagent un vocabulaire — saillance, décomposition, érosion
— mais aucun code qui doive rester synchronisé.
