"""Moteur de composition : plusieurs images, des critères, une image en sortie.

Reprend la logique de `blending-image2.py`, qui a produit les 230 œuvres des
séries de décembre 2024, avec deux changements de fond.

**La composition ne dépend plus de la résolution.** Les positions sont tirées
en coordonnées relatives, la saillance est toujours calculée à une taille de
référence fixe, et les forces des effets sont exprimées en fraction de la
toile. Une même recette rendue en 1024 px et en 6000 px donne la même image,
en plus fin. L'original tirait des positions en pixels sur une toile figée à
1024 : la composition n'existait qu'à cette taille.

**Le hasard est retenu, pas supprimé.** Chaque rendu part d'une graine. Sans
graine fournie, elle est tirée au sort : l'outil continue de produire des
images imprévues. Mais la graine sort avec l'image, ce qui permet de reprendre
un tirage retenu et de le recalculer en grand. C'est son seul usage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

# Taille à laquelle la saillance est toujours calculée. Le résidu spectral
# réagit à la résolution : calculé sur une image de 6000 px il trouve un détail
# plus fin que sur la même image en 1024. Le figer est ce qui garantit que la
# même recette donne la même composition à toutes les tailles.
SALIENCY_REFERENCE = 800

# Part de la toile occupée par le plus grand côté d'une image source.
# 800/1024 dans l'original, conservé pour retrouver le cadrage des séries.
SOURCE_SPAN = 800 / 1024

# Taille de rendu par défaut pour l'exploration
PREVIEW_SIZE = 1024

# Les quatre matières de blending-image2.py, plus les deux venues de la voie
# vidéo : rien n'obligeait à les tenir séparées.
EFFECTS = ("pixelate", "bitmap", "saturate", "halftone", "pixelsort", "shift",
           "mosh")


@dataclass(frozen=True)
class Effect:
    """Un effet et sa force, exprimée en fraction de la toile.

    `strength` vaut par exemple 0.008, soit 8 px sur une toile de 1024 et
    47 px sur une toile de 6000. Les forces en pixels de l'original donnaient
    un grain quatre fois plus fin dès qu'on quadruplait la taille.
    """

    name: str
    strength: float
    # La forme du point de trame. Sans effet pour les autres matières.
    shape: str = "round"
    # L'inclinaison, en degrés, pour les matières qui travaillent par lignes.
    # Propre à chaque matière : une trame à 45° sur un tri à l'horizontale
    # est une combinaison qu'un angle commun interdirait.
    angle: float = 0.0

    def pixels(self, canvas: int, minimum: int = 2) -> int:
        """Convertit la force en pixels pour une toile donnée."""
        return max(minimum, int(round(self.strength * canvas)))


@dataclass(frozen=True)
class Recipe:
    """Tout ce qu'il faut pour refaire exactement une image, à n'importe quelle taille."""

    sources: tuple[str, ...]
    seed: int
    # Matières appliquées à toutes les images qui n'ont pas de réglage propre
    effects: tuple[Effect, ...] = ()
    # Une entrée par image, dans l'ordre des sources : None pour suivre le
    # réglage commun, un tuple — même vide — pour le remplacer. Permet de
    # tramer une image, d'en pixelliser une autre, et d'en laisser une nette.
    per_image: tuple[tuple[Effect, ...] | None, ...] = ()
    # Part de l'image livrée aux matières, en pour-cent, des zones les moins
    # saillantes vers les plus saillantes. Exprimé en centile et non en valeur
    # brute : la carte de saillance est si concentrée — médiane 0,02 — qu'un
    # seuil brut au-delà de 30 sur 255 recouvrait déjà tout. Les neuf dixièmes
    # de la course du réglage ne servaient à rien.
    saliency_threshold: int = 60
    # Jusqu'où les matières débordent de la zone calme, de 0 à 1. À 0 elles
    # s'y tiennent ; à 1 elles couvrent toute l'image, sujet compris. Sans
    # elle, les matières ne mordent que sur la part de l'image qui pèse le
    # moins dans la composition, et tout ce qui ne transforme pas
    # radicalement — un tri, un décalage — s'y noie.
    bleed: float = 0.0
    # Le cadrage. Par défaut, chaque image est posée entière dans la toile,
    # ce qui laisse le fond noir autour — c'est la composition de 2024. En
    # plein cadre, elle est agrandie jusqu'à couvrir la toile et déborde : il
    # ne reste pas de fond, comme sur une capture de vidéo.
    full_frame: bool = False
    smoothness: float = 3.0
    edge_blur: int = 0
    # Le débordement uint8 de la saturation, préservé tel quel : seize œuvres
    # de serie1_grece/SATURATE en dépendent. Voir apply_saturation().
    saturation_overflow: bool = True

    def with_seed(self, seed: int) -> Recipe:
        return replace(self, seed=int(seed))


def effects_for(recipe: Recipe, index: int) -> tuple[Effect, ...]:
    """Les matières d'une image : les siennes, sinon celles de la recette."""
    if index < len(recipe.per_image) and recipe.per_image[index] is not None:
        return recipe.per_image[index]
    return recipe.effects


def new_seed() -> int:
    """Une graine au hasard, pour un tirage que personne n'a demandé."""
    return random.randrange(2**31)


# ---------------------------------------------------------------------------
# Saillance
# ---------------------------------------------------------------------------

def saliency_of(image: np.ndarray) -> np.ndarray:
    """Carte de saillance en niveaux de gris, calculée à taille de référence.

    Le calcul se fait toujours sur une version ramenée à SALIENCY_REFERENCE,
    puis la carte est agrandie. C'est ce qui rend la composition stable d'une
    taille de rendu à l'autre.
    """
    height, width = image.shape[:2]
    scale = SALIENCY_REFERENCE / max(height, width)
    if scale < 1.0:
        reference = cv2.resize(
            image, (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        reference = image

    try:
        detector = cv2.saliency.StaticSaliencySpectralResidual_create()
        found, raw = detector.computeSaliency(reference)
        if not found:
            raise RuntimeError("résidu spectral indisponible")
        card = (raw * 255).astype(np.uint8)
    except Exception:
        # Repli : contraste local. Une image doit toujours pouvoir être
        # composée, même si le module saliency d'OpenCV manque.
        grey = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, (21, 21), 0)
        card = cv2.normalize(
            cv2.absdiff(grey, blurred), None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    return cv2.resize(card, (width, height), interpolation=cv2.INTER_LINEAR)


def smooth_mask(mask: np.ndarray, smoothness: float, edge_blur: int) -> np.ndarray:
    """Adoucit la carte de saillance, et peut n'en garder que les bords.

    Le gradient morphologique demande un noyau d'au moins 3 : à 1, dilatation
    et érosion rendent la carte inchangée, leur différence est nulle partout,
    et l'image disparaissait entièrement de la composition. Le gradient d'une
    carte déjà lissée est par ailleurs de très faible amplitude — sur des
    photographies, un maximum de 13 sur 255 — donc invisible aussi. Il est
    ramené à toute la plage, sans quoi le réglage n'aurait jamais servi.
    """
    if smoothness > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), smoothness)
    if edge_blur > 0:
        largeur = max(3, int(edge_blur) | 1)
        kernel = np.ones((largeur, largeur), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, largeur / 3.0))
        crete = float(mask.max())
        if crete > 0:
            mask = np.clip(mask.astype(np.float32) * (255.0 / crete), 0, 255) \
                     .astype(mask.dtype)
    return mask


# ---------------------------------------------------------------------------
# Effets
# ---------------------------------------------------------------------------

def apply_pixelate(image: np.ndarray, mask: np.ndarray, block: int) -> np.ndarray:
    height, width = image.shape[:2]
    small = cv2.resize(
        image, (max(1, width // block), max(1, height // block)),
        interpolation=cv2.INTER_LINEAR,
    )
    blown = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(mask[:, :, None], blown, image)


def apply_bitmap(image: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, 128, 255, cv2.THRESH_BINARY)
    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    amount = float(np.clip(amount, 0.0, 1.0))
    blended = cv2.addWeighted(image, 1 - amount, binary, amount, 0)
    return np.where(mask[:, :, None], blended, image)


def apply_saturate(
    image: np.ndarray, mask: np.ndarray, factor: float, overflow: bool = True
) -> np.ndarray:
    """Sature les zones du masque.

    `overflow=True` reproduit le dépassement de l'original : la saturation
    était multipliée dans un tableau uint8, si bien qu'au-delà de 255 les
    valeurs repassaient par le bas (150 x 2 donnait 44) et cassaient les
    couleurs. Ce n'était pas voulu, mais seize œuvres en vivent. Le calcul est
    écrit explicitement plutôt que laissé au comportement de numpy, pour qu'une
    mise à jour ne le change jamais.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channel = hsv[:, :, 1].astype(np.int32)
    lifted = (channel * factor).astype(np.int32)
    lifted = np.mod(lifted, 256) if overflow else np.clip(lifted, 0, 255)
    hsv[:, :, 1] = np.where(mask, lifted, channel).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# Les six formes de trame de Photoshop, plus l'euclidienne, qui est la
# fonction de spot classique du PostScript.
HALFTONE_SHAPES = ("round", "square", "diamond", "line", "cross", "ellipse",
                   "euclidean")


def spot_field(cell: int, shape: str) -> np.ndarray:
    """Le champ d'une cellule de trame : 0 au centre, `cell/2` au bord.

    Une trame se ramène à un champ et à un seuil : on noircit là où le champ
    est sous le seuil, et le seuil suit la densité de la cellule. Changer de
    forme, c'est changer de champ — la mécanique reste la même.

    Seul l'ordre des valeurs compte : `rank_field` en tire la fraction de
    cellule à noircir, si bien qu'une forme n'a pas besoin d'être mise à
    l'échelle d'une autre.
    """
    axis = (np.arange(cell) - cell // 2).astype(np.float32)
    # Grille pleine : une forme qui n'utilise qu'un axe, comme la ligne,
    # doit quand même renvoyer une cellule carrée.
    x, y = np.meshgrid(axis, axis)
    half = max(1.0, cell / 2.0)

    if shape == "square":
        return np.maximum(np.abs(x), np.abs(y))
    if shape == "diamond":
        return np.abs(x) + np.abs(y)
    if shape == "line":
        return np.abs(y)
    if shape == "cross":
        # Sans le +1, le champ vaudrait 0 sur les deux axes médians et une
        # croix d'un pixel resterait visible dans les zones les plus claires.
        return np.minimum(np.abs(x), np.abs(y)) + 1.0
    if shape == "ellipse":
        # Aplatie : les points se rejoignent horizontalement avant de se
        # rejoindre verticalement, ce qui adoucit les dégradés.
        return np.sqrt((x * 0.75) ** 2 + (y * 1.33) ** 2)
    if shape == "euclidean":
        # Fonction de spot du PostScript. Le point est rond, devient carré à
        # mi-densité puis se creuse en rond inversé : la couverture suit le
        # niveau sans le saut de tonalité que produit un disque à 50 %.
        u, v = np.pi * x / half, np.pi * y / half
        return (2.0 - np.cos(u) - np.cos(v)) / 4.0 * half
    return np.sqrt(x ** 2 + y ** 2)


def rank_field(cell: int, shape: str) -> np.ndarray:
    """Le champ ramené à un rang dans [0, 1] : la fonction de spot.

    Chaque pixel de la cellule reçoit la fraction de pixels que la forme
    noircit avant lui. Noircir là où le rang est sous la densité couvre donc
    exactement cette densité, quelle que soit la forme — sans quoi un losange
    rendrait l'image trois fois plus sombre qu'un rond au même réglage.
    """
    field = spot_field(cell, shape)

    # Départage symétrique. Une forme à arêtes — carré, ligne, croix — a
    # beaucoup de pixels à valeur égale ; sans départage, `argsort` les
    # prendrait dans l'ordre de la mémoire et une cellule à demi remplie
    # monterait en escalier d'un seul côté. La distance au centre, pondérée
    # assez faiblement pour ne jamais franchir un écart du champ lui-même,
    # fait grandir la forme depuis son centre.
    axis = (np.arange(cell) - cell // 2).astype(np.float32)
    x, y = np.meshgrid(axis, axis)
    field = field + np.sqrt(x ** 2 + y ** 2) / (2.0 * max(1, cell))

    order = np.argsort(field, axis=None, kind="stable")
    rank = np.empty(field.size, dtype=np.float32)
    rank[order] = np.arange(field.size, dtype=np.float32) / max(1, field.size - 1)
    return rank.reshape(field.shape)


def apply_halftone(image: np.ndarray, mask: np.ndarray, cell: int,
                   shape: str = "round") -> np.ndarray:
    """Trame, vectorisée.

    L'original parcourait l'image en deux boucles Python avec un cv2.circle par
    cellule : tenable à 1024 px, environ 500 000 itérations à 6000 px. Ici
    tout se calcule en une passe numpy, ce qui rend le grand format possible.

    `shape` choisit la forme du point ; « round » reproduit l'original.
    """
    height, width = image.shape[:2]
    cell = max(2, int(cell))
    # Une forme inconnue — recette abîmée, nom mal orthographié — retombe sur
    # le rond, et sur son calcul d'origine, pas sur une approximation.
    shape = shape if shape in HALFTONE_SHAPES else "round"

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compléter jusqu'à un multiple de la cellule pour pouvoir replier le
    # tableau en grille de cellules
    pad_y = (-height) % cell
    pad_x = (-width) % cell
    padded = np.pad(grey, ((0, pad_y), (0, pad_x)), mode="edge")
    rows, cols = padded.shape[0] // cell, padded.shape[1] // cell

    # Moyenne par cellule
    means = padded.reshape(rows, cell, cols, cell).mean(axis=(1, 3))

    level = (255.0 - means) / 255.0

    if shape == "round":
        # Chemin d'origine, conservé tel quel : le rayon est tronqué à
        # l'entier et centré sur cell//2 comme le faisait le cv2.circle de
        # `blending-image2`, pour que la trame garde exactement son grain. Sa
        # couverture croît comme le carré de la densité, pas linéairement —
        # c'est un écart à la théorie, mais c'est ce qui fait les 230 œuvres.
        seuil = (level * (cell / 2.0)).astype(np.int32)
        champ = spot_field(cell, shape)
    else:
        # Les formes ajoutées suivent la densité linéairement, comme une
        # fonction de spot doit le faire.
        seuil = level.astype(np.float32)
        champ = rank_field(cell, shape)

    # Un point par cellule, par diffusion : (rows, 1, cols, 1) contre (cell, cell)
    inside = champ[None, :, None, :] <= seuil[:, None, :, None]
    dots = np.where(inside, 255, 0).astype(np.uint8)
    dots = dots.reshape(rows * cell, cols * cell)[:height, :width]

    toned = cv2.cvtColor(dots, cv2.COLOR_GRAY2BGR)
    return np.where(mask[:, :, None], toned, image)


def apply_effects(
    image: np.ndarray,
    saliency: np.ndarray,
    recipe: Recipe,
    canvas: int,
    effects: tuple[Effect, ...] | None = None,
) -> np.ndarray:
    """Applique la suite d'effets de la recette.

    Les effets mordent sur les zones **peu** saillantes, sauf la saturation qui
    fait l'inverse et rehausse ce qui attire l'œil. Cette asymétrie vient de
    l'original et fait l'image : le sujet reste net et se sature, le fond se
    décompose.
    """
    part = float(np.clip(recipe.saliency_threshold, 0, 100)) / 100.0
    # La bavure pousse le seuil vers le haut : à 1, la zone couvre tout.
    bave = float(np.clip(recipe.bleed, 0.0, 1.0))
    part = part + (1.0 - part) * bave
    quiet = saliency < np.quantile(saliency, part) if 0 < part < 1 \
        else (np.ones_like(saliency, bool) if part >= 1 else np.zeros_like(saliency, bool))
    result = image

    for effect in (recipe.effects if effects is None else effects):
        # Chaque matière porte son propre axe.
        from atelier.video import along_angle

        angle = float(effect.angle) % 180.0

        if effect.name == "pixelate":
            result = apply_pixelate(result, quiet, effect.pixels(canvas))
        elif effect.name == "bitmap":
            result = apply_bitmap(result, quiet, effect.strength)
        elif effect.name == "saturate":
            result = apply_saturate(
                result, ~quiet, effect.strength, recipe.saturation_overflow
            )
        elif effect.name == "halftone":
            cell = effect.pixels(canvas)
            result = along_angle(
                result, quiet, angle,
                lambda img, msk: apply_halftone(img, msk, cell, effect.shape),
            )
        elif effect.name == "pixelsort":
            # Importé ici : video importe engine, l'inverse au chargement
            # ferait un cycle.
            from atelier.video import pixel_sort

            low = effect.pixels(canvas, minimum=4)
            result = along_angle(
                result, quiet, angle,
                lambda img, msk: pixel_sort(
                    img, msk, "brightness", False, low, low * 4),
            )
        elif effect.name == "shift":
            from atelier.video import shift_channels

            result = shift_channels(
                result, effect.pixels(canvas, minimum=1),
                random.Random(f"{recipe.seed}:shift"),
                angle if effect.angle else None,
            )
        elif effect.name == "mosh":
            # Le tri canal par canal : c'est lui qui sépare les couleurs.
            #
            # `quirk=False` ici, contrairement à l'axe du temps. La
            # bizarrerie de l'original tire pour chaque canal une méthode de
            # tri, et le rouge ne tire qu'entre deux méthodes sans effet sur
            # un gris : il n'est donc jamais trié, et le vert comme le bleu
            # une fois sur deux. Une graine sur quatre ne triait rien du
            # tout. C'est tenable au fil d'une séquence, où les frames se
            # succèdent ; posé comme matière sur une image arrêtée, cela
            # donne un effet invisible. La séparation des couleurs tient de
            # toute façon aux longueurs et aux directions propres à chaque
            # canal, que `sort_channels_separately` conserve.
            from atelier.video import sort_channels_separately

            low = effect.pixels(canvas, minimum=4)
            result = along_angle(
                result, quiet, angle,
                lambda img, msk: sort_channels_separately(
                    img, msk, False, canvas,
                    random.Random(f"{recipe.seed}:mosh"),
                    np.random.default_rng(recipe.seed),
                    False, low, low * 4,
                ),
            )

    return result


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

@dataclass
class Placement:
    """Où une image atterrit, en fractions de la toile."""

    fx: float
    fy: float


def placements_for(recipe: Recipe) -> list[Placement]:
    """Tire la position de chaque image à partir de la seule graine.

    En fractions, jamais en pixels : c'est ce qui permet de rendre la même
    composition à n'importe quelle taille.
    """
    rng = random.Random(recipe.seed)
    return [Placement(rng.random(), rng.random()) for _ in recipe.sources]


def load_source(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"image illisible : {path}")
    return image


def layer_geometry(image: np.ndarray, placement, size: int, full_frame: bool):
    """Où et à quelle échelle une image atterrit sur la toile.

    Posée entière, elle tient dans une part de la toile et le fond reste
    visible autour. En plein cadre, elle est agrandie jusqu'à couvrir la
    toile ; elle déborde alors, et le placement choisit quelle part on garde.
    """
    height, width = image.shape[:2]
    scale = (size / min(height, width) if full_frame
             else (SOURCE_SPAN * size) / max(height, width))
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    # Un écart négatif — l'image est plus large que la toile — donne un
    # décalage négatif, donc un recadrage. C'est voulu.
    x = int(round(placement.fx * (size - target[0])))
    y = int(round(placement.fy * (size - target[1])))
    return target, x, y


def blend_into(canvas, weights, tile, weight, x: int, y: int) -> None:
    """Ajoute une tuile pondérée à la toile, en coupant ce qui en sort."""
    toile_h, toile_l = canvas.shape[:2]
    tuile_h, tuile_l = tile.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(toile_l, x + tuile_l), min(toile_h, y + tuile_h)
    if x0 >= x1 or y0 >= y1:
        return
    sx, sy = x0 - x, y0 - y
    part = tile[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    poids = weight[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    canvas[y0:y1, x0:x1] += part * poids[:, :, None]
    weights[y0:y1, x0:x1] += poids


def render(recipe: Recipe, size: int = PREVIEW_SIZE, sources=None) -> np.ndarray:
    """Rend la composition sur une toile carrée de `size` pixels.

    `sources` permet de passer des images déjà chargées, pour éviter de relire
    les fichiers à chaque mouvement de curseur.
    """
    images = sources if sources is not None else [load_source(p) for p in recipe.sources]
    if not images:
        raise ValueError("aucune image source")

    canvas = np.zeros((size, size, 3), dtype=np.float32)
    weights = np.zeros((size, size), dtype=np.float32)

    for index, (image, placement) in enumerate(zip(images, placements_for(recipe))):
        target, x, y = layer_geometry(image, placement, size, recipe.full_frame)
        resized = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
        saliency = smooth_mask(
            cv2.resize(saliency_of(image), target, interpolation=cv2.INTER_LINEAR),
            recipe.smoothness,
            recipe.edge_blur,
        )

        treated = apply_effects(
            resized, saliency, recipe, size, effects_for(recipe, index)
        )
        weight = saliency.astype(np.float32) / 255.0
        blend_into(canvas, weights, treated, weight, x, y)

    covered = weights > 0
    canvas[covered] /= weights[covered][:, None]
    return np.clip(canvas, 0, 255).astype(np.uint8)
