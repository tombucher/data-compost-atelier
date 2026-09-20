# Atelier

Instrument de composition par saillance et dégradation. Plusieurs images en
entrée, un axe de décomposition, une image ou une vidéo en sortie — à la
taille qu'on veut.

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

**Il n'y a qu'une œuvre, prise à des moments différents.** Les images se
superposent, puis le temps les défait. Une image fixe est un instant prélevé
sur cet axe, une vidéo est l'axe entier ; on ne choisit qu'au moment de
tirer. C'est ce qui fait tenir le propos : le compostage est un mécanisme qui
évolue dans le temps, pas deux modes d'export.

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
| Trame | Points de taille variable, façon similigravure. Sept formes, voir plus bas |
| Tri de pixels | Réordonne les pixels par bandes |
| Datamoshing | Trie chaque canal séparément : c'est lui qui sépare les couleurs |
| Décalage RVB | Décale le rouge et le bleu en sens opposés |

Les trois dernières viennent du datamoshing et ne vivaient que dans le temps.
Elles se posent aussi sur une image arrêtée, à l'instant 0 de l'axe.

Posé comme matière, le **datamoshing** trie les trois canaux. Dans l'axe du
temps il garde la bizarrerie de l'original, où le rouge n'est jamais trié et
le vert comme le bleu une fois sur deux : tenable au fil d'une séquence, mais
sur une image arrêtée une graine sur quatre ne triait rien du tout. La
séparation des couleurs tient de toute façon aux longueurs et aux directions
propres à chaque canal.

Les matières qui réordonnent sans changer les valeurs — les deux tris — ont
besoin de **bavure** pour se voir ; voir plus bas.

### L'axe des matières

La trame, les deux tris et le décalage de canaux travaillent par lignes, et
chacun porte **son propre axe**, réglable sous lui. Une trame ligne à 45° sur
un datamoshing à l'horizontale est une combinaison qu'un angle commun
interdirait — c'est d'ailleurs ainsi que fonctionne une trame couleur, où
chaque encre a le sien. Les autres matières n'ont pas d'axe et le réglage
n'apparaît pas.

L'image est tournée, la matière appliquée, puis l'image remise d'aplomb. La
double interpolation adoucit un peu le résultat, et ne se paie qu'en dehors
de l'horizontale.

Les forces marquées d'un grain — pixellisation, trame, tri, décalage — sont
exprimées en fraction de la toile, pas en pixels : elles gardent la même
allure à toutes les tailles. L'interface les affiche converties au format
visé.

### La saillance

Trois curseurs décident *où* les matières mordent.

**Part dégradée** dit quelle proportion de l'image leur revient, de la zone la
plus calme à la plus saillante. C'est un percentile : à 60, les 60 % les moins
saillants se décomposent. Exprimé en valeur brute de saillance, comme au
début, la moitié haute du curseur ne servait à rien — les cartes de saillance
sont très concentrées vers le bas.

**Lissage** floute la carte avant de s'en servir. Au minimum, les zones
dégradées ont des bords hachés ; en montant, elles se fondent dans le reste.

**Contours seuls** remplace la carte par son gradient : il ne reste que le
pourtour des zones saillantes, pas leur intérieur. L'image se réduit alors à
un liseré autour de son sujet, dont le centre disparaît de la composition.
À 0, le réglage est inactif.

**Bavure** dit de combien les matières débordent de la zone calme. Elle
existe parce que cette zone, qui couvre 60 % de la surface, ne porte que 14 %
du poids de la composition : les matières y mordaient sans se voir. Une trame
s'en sortait — elle passe l'image en noir et blanc — mais un tri de pixels ou
un tri par canaux, qui réordonnent sans changer les valeurs, se noyaient au
point de paraître sans effet. À 1, les matières couvrent toute l'image,
sujet compris.

### Le cadrage

Deux boutons en haut du panneau.

**Dans le cadre** pose chaque image entière dans la toile, ce qui laisse le
fond noir visible autour. C'est la composition de 2024 et le réglage par
défaut.

**Plein cadre** l'agrandit jusqu'à couvrir la toile : elle déborde, le
placement décide quelle part on garde, et il ne reste aucun fond — comme sur
une capture de vidéo.

### Les formes de trame

Sept formes, les six de Photoshop plus l'euclidienne du PostScript. Le menu
apparaît sous le curseur de trame quand elle est active.

| Forme | Ce qu'elle donne |
|---|---|
| Ronde | Le point d'offset. C'est la trame de l'original et celle des 230 œuvres |
| Carrée | Dure, franche, proche de la sérigraphie |
| Losange | Les points se rejoignent en diagonale |
| Ligne | Des bandes horizontales d'épaisseur variable, façon gravure |
| Croix | Une croix qui grossit jusqu'à fermer la cellule |
| Ellipse | Aplatie : les points se chaînent horizontalement avant de se rejoindre en hauteur, ce qui adoucit les dégradés |
| Euclidienne | Ronde, puis carrée à mi-densité, puis ronde en creux. C'est la fonction de spot du PostScript, qui évite le saut de tonalité à 50 % |

**La ronde rend plus clair que les autres, au même réglage.** Ce n'est pas un
défaut d'affichage. L'original calculait le rayon du point proportionnellement
à la densité, si bien que sa surface — donc son noir — croît comme le *carré*
de la densité : à mi-densité il ne couvre que 20 % de la cellule au lieu de
50 %. Les six formes ajoutées suivent la densité linéairement, comme doit le
faire une fonction de spot. Corriger la ronde changerait les 230 œuvres ;
elle garde donc son calcul, et l'écart est assumé.

Sous cinq ou six pixels de cellule, les formes cessent de se distinguer :
neuf pixels ne suffisent pas à séparer un carré d'un losange. Un grain très
fin annule donc le choix de la forme.

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

Les 230 œuvres n'ont que de la trame ronde — l'original ne savait pas en
faire d'autre. Le tirage la garde donc très majoritaire et ne laisse passer
une autre forme qu'une fois sur cinq.

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

## L'axe du temps

La barre sous l'aperçu va de **intact** à **composté**. À gauche, la
composition telle que la rendait `blending-image2` — bit pour bit, c'est
vérifié par un test. En avançant, chaque couche se trie par segments, perd
l'alignement de ses canaux, s'arrache par plaques et laisse voir celle du
dessous.

Chaque instant se calcule seul, sans rejouer ce qui précède : on se déplace
librement dans le temps. **Tirer cet instant** sort l'image où l'on est, à la
taille choisie ; **Tirer tout l'axe en vidéo** écrit un mp4. Le calcul d'une
vidéo dure des minutes en grand format, il tourne en tâche de fond et
l'avancement s'affiche.

C'est ce qui remplace la capture d'écran. Les 116 images de
`export-vid_sauv/SCREENSHOT` font 2254 px alors que leurs vidéos sources en
contenaient 3016 : la capture perdait un quart de la définition. Ici l'instant
sort à 4000 ou 8000 px.

### Les réglages de l'axe

| Réglage | Ce qu'il fait |
|---|---|
| Durée | La longueur de l'axe, en images. Elle ne dépend pas du nombre d'images choisies : une image de plus est une couche, pas une diapositive. |
| Décalage entre couches | À 0 tout pourrit ensemble. Au maximum, les couches s'effacent l'une après l'autre. Ce qui est posé en dernier part en premier. |
| Ce qui résiste | La part la plus dense de chaque couche, encore là à la fin. À 0, l'axe s'éteint au noir. |
| Segment le plus court / le plus long | La longueur des bandes triées. Les rapports entre canaux sont conservés — c'est leur écart qui sépare les couleurs, pas leur valeur. |
| Allongement | De combien les segments s'étirent à mesure que l'image se défait. |
| Décalage RVB | La séparation des canaux rouge et bleu, qui croît avec le temps. |

Ces réglages ne mordent qu'une fois l'axe engagé : à l'instant 1 la
composition est intacte, et les bouger n'y change rien. Le panneau le dit.

Les quatre bascules de **Fidélité** disent quoi garder de l'original : le
tri par canaux séparés, le rouge jamais trié, le tri sur la zone visible, la
dissolution par quantiles. Les trois dernières sont des corrections — voir
plus bas — et se décochent pour retrouver le rendu de 2025.

### Où sont passées les explications

Chaque réglage porte un petit `?` : le survol du libellé donne une phrase sur
ce qu'il fait concrètement à l'image. Les explications étaient auparavant
posées sous les curseurs, où elles se lisaient comme l'introduction du bloc
suivant. Seules les notes qui présentent une section entière restent à
l'écran, sous leur titre et marquées d'un trait.

### Pourquoi l'original ne montrait pas de datamoshing

Deux corrections sont nées de cette recherche, toutes deux débrayables.

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
