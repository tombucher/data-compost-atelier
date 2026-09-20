# Atelier

Un instrument pour composer plusieurs photographies en une seule image, puis
la laisser se décomposer.

## Ce que ça fait

On choisit deux à quatre images. Elles se superposent sur une toile carrée,
chacune n'apparaissant que là où elle retient le regard : le programme calcule
une **carte de saillance** — les zones qui attirent l'œil — et s'en sert comme
d'un pochoir. Le sujet de chaque photographie ressort, son fond s'efface au
profit des autres.

Sur cette composition on pose des **matières** : une trame de journal, un
passage en noir et blanc, une pixellisation, un tri des pixels par bandes, une
séparation des canaux de couleur. Elles mordent sur les zones les plus calmes
et laissent le sujet net. Cette asymétrie fait l'image.

Puis vient le temps. Une barre sous l'aperçu va de **intact** à **composté**.
En la parcourant, chaque image de la pile se défait : sa matière s'intensifie,
ses pixels se trient, ses couleurs se désalignent, et elle s'efface par
plaques en laissant voir celle du dessous. Ce qui est posé en dernier part le
premier ; ce qui est le plus dense résiste le plus longtemps.

## Ce qu'on en sort

Un instant de cet axe, en image fixe, à la taille qu'on veut — 8000 px, soit
68 cm à 300 dpi. Ou bien l'axe entier, en vidéo. Le choix ne se fait qu'au
moment d'exporter : il n'y a pas deux modes, il y a une œuvre et des moments
où la prendre.

Chaque image exportée porte dans ses métadonnées de quoi se refaire — la
graine du tirage et tous les réglages. Pas de fichier de projet à côté, pas de
bibliothèque à entretenir. Ce qu'on n'exporte pas disparaît.

La composition ne dépend pas de la résolution : les forces sont exprimées en
fraction de la toile, jamais en pixels. On cherche en basse définition, où
l'aperçu se recalcule en continu, et le tirage retenu se relance en grand sans
que le grain change d'échelle.

## Démarrer

Il faut Python — le développement s'est fait en 3.14 — et un dossier d'images
à soi : le dépôt n'en fournit aucune.

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python demarrer.py ~/Documents/mes-images
```

Le dernier argument est le dossier où prendre les images. Le navigateur
s'ouvre sur l'atelier. Rien n'est hébergé : le serveur tourne en local, les
images ne quittent pas le disque.

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
se superposer. À droite, activer les matières et régler. L'aperçu se recalcule
en continu : sur trois images, comptez 0,2 s pour une composition simple, 0,6 s
avec un tri, et environ 1 s pour un instant pris au milieu de l'axe.

Quand une composition tient, choisir un format et **Tirer**. Le fichier part
dans `sorties/`, avec sa recette à l'intérieur.

Un tirage déjà fait réapparaît sous **Reprendre** : un clic restaure ses
images, sa graine, son instant sur l'axe et tous ses réglages. C'est le seul
usage de la graine — elle sert à revenir sur un tirage pour l'agrandir, pas à
répéter quoi que ce soit.

Le menu en haut à gauche change de dossier sans relancer l'atelier : il
propose le dossier parent et les dossiers voisins qui contiennent des images,
en signalant celui qui reçoit les tirages.

Au-dessus des matières, un second menu choisit **à quelle image** elles
s'appliquent. Par défaut « Toutes les images » ; en choisissant une image
précise, les curseurs ne règlent plus qu'elle — on peut tramer l'une,
pixelliser l'autre, et laisser la troisième intacte. Les images ayant leur
propre réglage sont marquées d'un trait dans la liste.

Chaque réglage porte un `?` en bout de ligne. Un clic ouvre une phrase sur ce
qu'il fait concrètement à l'image ; un clic ailleurs, ou Échap, la referme.

## Les matières

Les matières mordent sur les zones **peu** saillantes, sauf la saturation qui
fait l'inverse et rehausse ce qui attire l'œil. Cette asymétrie fait l'image :
le sujet reste net et se sature, le fond se décompose.

| Matière | Ce qu'elle fait |
|---|---|
| Pixellisation | Remplace des blocs carrés par leur couleur moyenne |
| Bitmap | Bascule en noir et blanc purs, mélangés à l'image à la dose choisie |
| Saturation | Rehausse les couleurs du sujet |
| Trame | Points de taille variable, façon similigravure. Sept formes, voir plus bas |
| Tri de pixels | Réordonne les pixels par bandes, du sombre au clair |
| Datamoshing | Trie chaque canal séparément : c'est lui qui sépare les couleurs |
| Décalage RVB | Décale le rouge et le bleu en sens opposés |

Les forces de la pixellisation, de la trame, des deux tris et du décalage sont
exprimées en fraction de la toile. L'interface les affiche converties au
format de tirage visé.

Les deux tris réordonnent les pixels sans changer leurs valeurs. Ils ont donc
besoin de **bavure** pour se voir — voir plus bas.

Le **datamoshing** donne à chaque canal sa propre longueur de segment et sa
propre direction : c'est de là que vient la séparation des couleurs. Une case
**croiser les canaux**, sous lui, envoie le vert perpendiculairement au rouge
et au bleu. Décochée, les coulées suivent un seul sens.

### L'axe des matières

La trame, les deux tris et le décalage de canaux travaillent par lignes, et
chacun porte **son propre axe**, réglable sous lui. Une trame ligne à 45° sur
un datamoshing à l'horizontale est une combinaison qu'un angle commun
interdirait — c'est d'ailleurs ainsi que fonctionne une trame couleur, où
chaque encre a le sien. Les autres matières n'ont pas d'axe, et le réglage
n'apparaît pas.

L'image est tournée, la matière appliquée, puis l'image remise d'aplomb. La
double interpolation adoucit un peu le résultat, et ne se paie qu'en dehors
de l'horizontale.

### Les formes de trame

Sept formes : les six de Photoshop, plus l'euclidienne du PostScript. Le menu
apparaît sous le curseur de trame quand elle est active.

| Forme | Ce qu'elle donne |
|---|---|
| Ronde | Le point d'offset, le plus courant |
| Carrée | Dure, franche, proche de la sérigraphie |
| Losange | Les points se rejoignent en diagonale |
| Ligne | Des bandes d'épaisseur variable, façon gravure |
| Croix | Une croix qui grossit jusqu'à fermer la cellule |
| Ellipse | Aplatie : les points se chaînent horizontalement avant de se rejoindre en hauteur, ce qui adoucit les dégradés |
| Euclidienne | Ronde, puis carrée à mi-densité, puis ronde en creux. C'est la fonction de spot du PostScript, qui évite le saut de tonalité à 50 % |

**La ronde rend plus clair que les autres, au même réglage.** Son rayon suit
la densité, donc sa surface — son noir — croît comme le *carré* de celle-ci :
à mi-densité elle ne couvre que 20 % de la cellule au lieu de 50 %. Les six
autres suivent la densité linéairement, comme le fait une fonction de spot.
L'écart est voulu : la ronde est la trame de référence de l'outil, et son
calcul n'est pas modifié.

Sous cinq ou six pixels de cellule, les formes cessent de se distinguer : neuf
pixels ne suffisent pas à séparer un carré d'un losange. Un grain très fin
annule donc le choix de la forme.

## La saillance

Quatre curseurs décident *où* les matières mordent.

**Part dégradée** dit quelle proportion de l'image leur revient, de la zone la
plus calme à la plus saillante. C'est un percentile : à 60, les 60 % les moins
saillants se décomposent.

**Lissage** floute la carte avant de s'en servir. Au minimum, les zones
dégradées ont des bords hachés ; en montant, elles se fondent dans le reste.

**Contours seuls** remplace la carte par son gradient : il ne reste que le
pourtour des zones saillantes, pas leur intérieur. L'image se réduit alors à
un liseré autour de son sujet, dont le centre disparaît de la composition.
À 0, le réglage est inactif.

**Bavure** dit de combien les matières débordent de la zone calme. Ce réglage
compte plus qu'il n'y paraît : la zone calme couvre 60 % de la surface mais ne
porte que 14 % du poids de la composition, puisque le poids de chaque pixel
est sa saillance. Une trame s'y voit — elle passe l'image en noir et blanc —
mais un tri, qui réordonne sans changer les valeurs, s'y noie. La bavure lui
donne de quoi mordre. À 1, les matières couvrent toute l'image, sujet compris.

## Le cadrage

Deux boutons en haut du panneau.

**Dans le cadre** pose chaque image entière dans la toile, ce qui laisse le
fond noir visible autour. C'est le réglage par défaut.

**Plein cadre** l'agrandit jusqu'à couvrir la toile : elle déborde, le
placement décide quelle part on garde, et il ne reste aucun fond.

## L'axe du temps

La barre sous l'aperçu va de **intact** à **composté**. À gauche, la
composition telle qu'on l'a réglée. En avançant, chaque couche se trie par
segments, perd l'alignement de ses canaux, s'arrache par plaques et laisse
voir celle du dessous.

Chaque instant se calcule seul, sans rejouer ce qui précède : on se déplace
librement dans le temps. **Tirer cet instant** sort l'image où l'on est, à la
taille choisie ; **Tirer tout l'axe en vidéo** écrit un mp4. Le calcul d'une
vidéo se compte en minutes dès qu'on vise le grand format ; il tourne en
tâche de fond et l'avancement s'affiche.

Une image tirée de l'axe sort à la définition du rendu, pas à celle de
l'écran : c'est ce qui remplace la capture d'écran sur une vidéo, qui plafonne
à la taille du moniteur.

### Ce que le temps travaille

Le compost décompose la matière qui est là, pas une matière générique. Les
matières choisies **mûrissent** donc le long de l'axe : la trame grossit, la
pixellisation s'élargit, le bitmap durcit vers le noir et blanc francs, la
saturation monte, les tris s'allongent. Chaque matière mûrit selon sa
nature — le bitmap plafonne à 1, la saturation grandit depuis 1.

Par-dessus vient la **déchirure** : le tri canal par canal et le décalage de
canaux. Elle est la même pour toutes les matières, et c'est voulu, car c'est
le geste de la décomposition elle-même. Elle se dose, et s'éteint à zéro — le
temps ne fait alors plus que mûrir ce qu'on a posé, et évider.

### Les réglages de l'axe

| Réglage | Ce qu'il fait |
|---|---|
| Durée | La longueur de l'axe, en images. Elle ne dépend pas du nombre d'images choisies : une image de plus est une couche, pas une diapositive. |
| Mûrissement | De combien les matières posées s'intensifient au bout de l'axe. C'est ce qui lie la décomposition à ce qu'on a choisi. |
| Déchirure | La part qu'y prend le tri par canaux et le décalage RVB. À zéro, le temps ne fait plus que mûrir et évider. |
| Décalage entre couches | À 0 tout pourrit ensemble. Au maximum, les couches s'effacent l'une après l'autre. Ce qui est posé en dernier part en premier. |
| Ce qui résiste | La part la plus dense de chaque couche, encore là à la fin. À 0, l'axe s'éteint au noir. |
| Déchirure la plus courte / la plus longue | La longueur des bandes triées. Les rapports entre canaux sont conservés : c'est leur écart qui sépare les couleurs, pas leur valeur. |
| Allongement | De combien les bandes s'étirent à mesure que l'image se défait. |
| Décalage RVB | La séparation des canaux rouge et bleu, qui croît avec le temps. |

Ces réglages ne mordent qu'une fois l'axe engagé : au premier instant la
composition est intacte, et les bouger n'y change rien. Le panneau le dit.

## Au hasard

Le bouton tire une composition entière — images, matières, réglages — et le
menu voisin restreint le tirage à une veine.

Le tirage n'est pas uniforme. Il suit les proportions d'un corpus de
référence, mesuré sur quatre grandeurs : contraste, saturation, part de noir
et blanc purs, énergie haute fréquence. Il en ressort une répartition —
trame 53 %, bitmap 23 %, pixellisation 17 %, saturation 7 % — et les forces de
chaque veine sont réglées pour retomber sur les mesures de sa famille. Deux
plages viennent directement de ces chiffres : le bitmap ne descend pas sous
0,85, faute de quoi le seuillage reste mêlé à l'image et le noir et blanc
francs n'apparaît pas ; et le débordement de saturation ne se tire qu'une fois
sur sept.

La trame ronde reste très majoritaire, et le tirage n'incline une matière
qu'une fois sur quatre. Les écarts que le tirage conserve avec son corpus de
référence, et pourquoi, sont écrits dans `atelier/chance.py`.

## Fidélité

Cinq cases regroupent des comportements de rendu inhabituels, qu'on peut
vouloir ou non. Ils viennent des scripts dont l'outil est issu, où certains
étaient des accidents ; ils sont conservés parce qu'ils font l'image, et
réglables pour qu'on puisse s'en passer.

**Débordement de saturation.** La saturation est multipliée dans un tableau
d'entiers 8 bits : au-delà de 255, les valeurs repassent par le bas — 150 × 2
donne 44 — et les couleurs cassent. Le calcul est écrit explicitement pour
qu'aucune mise à jour de numpy ne le change. Décoché, la saturation plafonne
normalement à 255.

**Trier les canaux séparément.** Chaque canal reçoit sa propre longueur de
segment et sa propre méthode de tri, ce qui sépare les couleurs. Décoché, les
trois canaux se trient ensemble et l'image reste grise.

**Laisser le rouge intact.** Chaque canal est répliqué en gris avant d'être
trié. Or trier un gris par teinte ou par saturation ne change rien : ces
valeurs y sont toutes nulles. Comme le rouge ne tire qu'entre ces deux
méthodes, il n'est jamais trié, et le vert comme le bleu une fois sur deux.
C'est cette asymétrie qui donne sa dominante à l'effet. Décoché, les trois
canaux sont traités pareil.

**Trier ce qu'on voit.** Le tri porte sur la zone encore visible. Décoché, il
porte sur la zone en train de disparaître — il ne se voit alors presque pas.

**Dissolution régulière.** Les seuils de dissolution suivent les quantiles de
la carte de saillance, de sorte que chaque image de la séquence retire une
part comparable. Décoché, ils sont répartis de 0 à 1 sur les valeurs brutes.
Comme la carte de saillance est très concentrée vers le bas — médiane 0,02,
moyenne 0,04 sur des photographies —, l'image se vide alors en deux ou trois
images et ne bouge plus ensuite :

```
seuils par quantiles : 100 100 87 87 75 75 67 61 54 49 ... 0
seuils bruts         : 100  16  6  3  2  1  1  0  0  0 ... 0
```

## Tests

```bash
venv/bin/python -m pytest tests/ -q
```

Une partie des tests demande des photographies dans `compost-test/` pour
mesurer des amplitudes ; ils se sautent proprement si le dossier est absent.

`tests/test_fidelity.py` compare le moteur à des implémentations de référence
recopiées telles quelles : moins de 2 % de pixels différents pour la trame,
identique à l'octet près pour la saturation. Le reste vérifie qu'une graine
donne la même composition à toutes les tailles, que chaque instant de l'axe se
calcule seul, et que la recette survit à l'aller-retour dans l'image.

## Limites connues

- Quand la cellule de trame ne divise pas la toile, la dernière cellule
  partielle peut rester sombre : son point est centré dans la cellule
  complétée, dont le centre peut tomber hors de l'image. Borné à une cellule.
- Une vidéo est longue à calculer. Le coût d'une image croît à peu près comme
  la taille puissance 1,4 : environ 1 s en 900 px, 6 s en 3000 px. Une
  séquence de 150 images en 3000 px demande donc un quart d'heure. Elle tourne
  en tâche de fond, mais il n'y a pas de rendu partiel ni de reprise.
- L'interface n'a été éprouvée que sur sept images sources. La colonne de
  vignettes n'a pas été essayée sur un dossier fourni.
- Les matières s'appliquent à chaque image avant son placement sur la toile :
  une coulée s'arrête donc au bord de sa source et ne bave pas sur les
  voisines. Le plein cadre supprime le fond noir, mais pas cette limite.

## Le projet

L'atelier prolonge un travail d'art numérique sur le **compostage de
données** : traiter des fichiers comme de la matière organique, qui se
décompose et nourrit autre chose.
[data-compost](https://github.com/tombucher/data-compost) en est
l'installation, une machine qui tourne seule sur Raspberry Pi, sous contrainte
de robustesse et de matériel. L'atelier en est l'instrument, qu'on pilote,
sous contrainte de vitesse d'essai et de qualité de sortie. Les deux partagent
un vocabulaire — saillance, décomposition, érosion — mais aucune ligne de code
qui doive rester synchronisée, d'où deux dépôts.
