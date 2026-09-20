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

Le plus court chemin : **Au hasard**, en haut à droite. Le bouton tire les
images et tous les réglages, et l'aperçu apparaît. Rien n'y est figé : tout
ce qu'il a posé se reprend à la main ensuite. Le menu à côté restreint le
tirage à une veine — trame, bitmap, pixellisation ou saturation.

Sinon, à la main : à gauche, cliquer les images dans l'ordre où elles doivent
se superposer. À droite, activer les matières et régler. L'aperçu se recalcule en continu —
environ 150 ms sur trois images.

Quand une composition tient, choisir un format et **Tirer**. Le fichier part
dans `sorties/`, avec sa recette à l'intérieur.

Un tirage déjà fait réapparaît sous **Reprendre** : un clic restaure ses
images, sa graine et tous ses réglages. C'est le seul usage de la graine.

Le menu en haut à gauche change de série sans relancer l'atelier : il propose
le dossier parent et les dossiers voisins qui contiennent des images.

Au-dessus des matières, un second menu choisit **à quelle image** elles
s'appliquent. Par défaut « Toutes les images » ; en choisissant une image
précise, les curseurs ne règlent plus qu'elle — on peut tramer l'une,
pixelliser l'autre, et laisser la troisième intacte. Les images ayant leur
propre réglage sont marquées d'un trait dans la liste.

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
| Tri de pixels | Réordonne les pixels par bandes, venu de la voie vidéo |
| Décalage RVB | Sépare les canaux de couleur, venu de la voie vidéo |

Les forces marquées d'un grain — pixellisation, trame, tri, décalage — sont
exprimées en fraction de la toile, pas en pixels : elles gardent la même
allure à toutes les tailles. L'interface les affiche converties au format
visé.

Le curseur **Part dégradée** dit quelle proportion de l'image revient aux
matières, de la zone la plus calme à la plus saillante. C'est un percentile :
à 60, les 60 % les moins saillants se décomposent. Exprimé en valeur brute de
saillance, comme au début, la moitié haute du curseur ne servait à rien — les
cartes de saillance sont très concentrées vers le bas.

### Au hasard

Le tirage ne choisit pas uniformément. Les 230 œuvres de 2024 ont été mesurées
sur quatre grandeurs — contraste, saturation, part de noir et blanc purs,
énergie haute fréquence — et les deux séries que Tom avait rangées par
technique ont servi à classer les autres. Il en ressort une répartition :
trame 53 %, bitmap 23 %, pixellisation 17 %, saturation 7 %. Le tirage suit
ces parts, et les forces de chaque veine sont réglées pour retomber sur les
mesures de sa famille.

Deux réglages viennent directement des chiffres : le bitmap ne descend pas
sous 0,85, sinon le seuillage reste mêlé à l'image et le noir et blanc francs
n'apparaît pas ; et le débordement de saturation, qui ne concerne que 16
œuvres sur 230, ne se tire qu'une fois sur sept.

Ce qui reste en écart et pourquoi, c'est écrit dans `atelier/chance.py`.

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

Bascule **Vidéo** en haut à droite. Les deux voies partagent tout : les
matières de l'image s'appliquent à chaque image de la séquence, réglages par
image compris, et le tri de pixels comme le décalage de canaux — nés du
datamoshing — sont disponibles sur une image fixe.

Les mêmes images et les mêmes matières,
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

### Pourquoi l'original ne montrait pas de datamoshing

Deux raisons, trouvées en cherchant l'effet manquant.

**Le tri tombait dans la zone cachée.** L'image 1 était triée là où son masque
valait 0, puis composée avec le poids de ce même masque : les deux zones étant
complémentaires, le tri portait exactement sur ce qui n'était pas affiché. Le
commentaire de `datamoshing4.py` annonçait pourtant « img1 reste visible et se
dégrade ». L'intention et le code se contredisaient.

**La dissolution se jouait en deux images.** Le résidu spectral produit une
carte très concentrée — sur les photographies du projet, médiane 0,02 et
moyenne 0,04. Des seuils répartis de 0 à 1 donnaient donc :

```
part conservée : 100  16  6  3  2  1  1  0  0  0 ... 0
```

Tout basculait entre la première et la troisième image, et les vingt suivantes
étaient figées. Les seuils suivent maintenant les quantiles de la carte :

```
part conservée : 100 100 87 87 75 75 67 61 54 49 ... 0
```

Les deux comportements d'origine restent accessibles par `Motion.sort_visible`
et `Motion.even_dissolve`.

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
