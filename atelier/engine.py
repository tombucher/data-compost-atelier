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

EFFECTS = ("pixelate", "bitmap", "saturate", "halftone")


@dataclass(frozen=True)
class Effect:
    """Un effet et sa force, exprimée en fraction de la toile.

    `strength` vaut par exemple 0.008, soit 8 px sur une toile de 1024 et
    47 px sur une toile de 6000. Les forces en pixels de l'original donnaient
    un grain quatre fois plus fin dès qu'on quadruplait la taille.
    """

    name: str
    strength: float

    def pixels(self, canvas: int, minimum: int = 2) -> int:
        """Convertit la force en pixels pour une toile donnée."""
        return max(minimum, int(round(self.strength * canvas)))


@dataclass(frozen=True)
class Recipe:
    """Tout ce qu'il faut pour refaire exactement une image, à n'importe quelle taille."""

    sources: tuple[str, ...]
    seed: int
    effects: tuple[Effect, ...] = ()
    saliency_threshold: int = 120
    smoothness: float = 3.0
    edge_blur: int = 0
    # Le débordement uint8 de la saturation, préservé tel quel : seize œuvres
    # de serie1_grece/SATURATE en dépendent. Voir apply_saturation().
    saturation_overflow: bool = True

    def with_seed(self, seed: int) -> Recipe:
        return replace(self, seed=int(seed))


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
    """Adoucit la carte de saillance, et peut n'en garder que les bords."""
    if smoothness > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), smoothness)
    if edge_blur > 0:
        kernel = np.ones((edge_blur, edge_blur), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        mask = cv2.GaussianBlur(mask, (0, 0), edge_blur)
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


def apply_halftone(image: np.ndarray, mask: np.ndarray, cell: int) -> np.ndarray:
    """Trame de points, vectorisée.

    L'original parcourait l'image en deux boucles Python avec un cv2.circle par
    cellule : tenable à 1024 px, environ 500 000 itérations à 6000 px. Ici
    tout se calcule en une passe numpy, ce qui rend le grand format possible.
    """
    height, width = image.shape[:2]
    cell = max(2, int(cell))

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compléter jusqu'à un multiple de la cellule pour pouvoir replier le
    # tableau en grille de cellules
    pad_y = (-height) % cell
    pad_x = (-width) % cell
    padded = np.pad(grey, ((0, pad_y), (0, pad_x)), mode="edge")
    rows, cols = padded.shape[0] // cell, padded.shape[1] // cell

    # Moyenne par cellule
    means = padded.reshape(rows, cell, cols, cell).mean(axis=(1, 3))

    # Rayon du point : une cellule sombre donne un gros point. Tronqué à
    # l'entier et centré sur cell//2 comme le faisait le cv2.circle d'origine,
    # pour que la trame garde exactement le même grain.
    radius = ((255.0 - means) / 255.0 * (cell / 2.0)).astype(np.int32)

    # Distance au centre de la cellule, identique pour toutes
    axis = np.arange(cell) - cell // 2
    distance = np.sqrt(axis[:, None] ** 2 + axis[None, :] ** 2)

    # Un point par cellule, par diffusion : (rows, 1, cols, 1) contre (cell, cell)
    inside = distance[None, :, None, :] <= radius[:, None, :, None]
    dots = np.where(inside, 255, 0).astype(np.uint8)
    dots = dots.reshape(rows * cell, cols * cell)[:height, :width]

    toned = cv2.cvtColor(dots, cv2.COLOR_GRAY2BGR)
    return np.where(mask[:, :, None], toned, image)


def apply_effects(
    image: np.ndarray,
    saliency: np.ndarray,
    recipe: Recipe,
    canvas: int,
) -> np.ndarray:
    """Applique la suite d'effets de la recette.

    Les effets mordent sur les zones **peu** saillantes, sauf la saturation qui
    fait l'inverse et rehausse ce qui attire l'œil. Cette asymétrie vient de
    l'original et fait l'image : le sujet reste net et se sature, le fond se
    décompose.
    """
    quiet = saliency < recipe.saliency_threshold
    result = image

    for effect in recipe.effects:
        if effect.name == "pixelate":
            result = apply_pixelate(result, quiet, effect.pixels(canvas))
        elif effect.name == "bitmap":
            result = apply_bitmap(result, quiet, effect.strength)
        elif effect.name == "saturate":
            result = apply_saturate(
                result, ~quiet, effect.strength, recipe.saturation_overflow
            )
        elif effect.name == "halftone":
            result = apply_halftone(result, quiet, effect.pixels(canvas))

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
    span = max(1, int(round(SOURCE_SPAN * size)))

    for image, placement in zip(images, placements_for(recipe)):
        height, width = image.shape[:2]
        scale = span / max(height, width)
        target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))

        resized = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
        saliency = smooth_mask(
            cv2.resize(saliency_of(image), target, interpolation=cv2.INTER_LINEAR),
            recipe.smoothness,
            recipe.edge_blur,
        )

        # Position en pixels, déduite de la fraction : même cadrage à toute taille
        x = int(round(placement.fx * max(0, size - target[0])))
        y = int(round(placement.fy * max(0, size - target[1])))

        treated = apply_effects(resized, saliency, recipe, size)
        weight = saliency.astype(np.float32) / 255.0

        canvas[y:y + target[1], x:x + target[0]] += treated * weight[:, :, None]
        weights[y:y + target[1], x:x + target[0]] += weight

    covered = weights > 0
    canvas[covered] /= weights[covered][:, None]
    return np.clip(canvas, 0, 255).astype(np.uint8)
