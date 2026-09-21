"""Le temps du compost : une composition unique, qui se décompose.

Il n'y a plus deux voies. Il y a une composition — les images superposées,
pondérées par leur saillance — et un axe de temps qui la défait. L'instant 0
donne exactement ce que rendait la voie image ; en avançant, chaque couche se
trie, se déchire, perd ses canaux, et s'efface en laissant voir celle du
dessous. Une image fixe est un instant prélevé sur cet axe, une vidéo est
l'axe entier. Le choix ne se fait qu'à l'export.

## Pourquoi les couches ne partent pas ensemble

Ce qui est posé en dernier est le plus exposé, et se décompose en premier.
`stagger` règle ce décalage : à 0 toutes les couches pourrissent de concert,
à 1 elles s'effacent strictement l'une après l'autre. Entre les deux, les
décompositions se chevauchent, ce qui est le cas intéressant.

## Pourquoi il reste toujours quelque chose

Un compost ne finit pas sur du vide, il finit sur de l'humus. `residue`
garde la part la plus dense de chaque couche : à la fin de l'axe, ce que la
saillance désignait comme le sujet est encore là, seul, entouré de ce qui
s'est défait. Mettre `residue` à 0 laisse la séquence s'éteindre au noir.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

import cv2
import numpy as np

from atelier.engine import (
    BLEED_SOFTNESS,
    Recipe,
    apply_effects,
    blend_into,
    effects_for,
    layer_geometry,
    load_source,
    normaliser,
    placements_for,
    saliency_of,
    smooth_mask,
)
from atelier.video import (
    dissolve_field,
    frame_draw,
    pixel_sort,
    retention_at,
    shift_channels,
    sort_channels_separately,
)


@dataclass(frozen=True)
class Decay:
    """L'axe du temps, et la manière dont la matière se défait.

    Les longueurs sont en fraction du côté de la toile : `min_segment` à 0.03
    vaut 30 px sur 1024 et 90 px sur 3000, donc le même geste à toute taille.
    """

    frames: int = 48
    fps: int = 24

    # Deux manières de faire passer le temps sur les mêmes images.
    #
    # Superposées, c'est la pile : toutes les images sont là en même temps,
    # pondérées par leur saillance, et le temps défait l'ensemble. Ce que
    # l'évidement creuse finit par déboucher sur le fond.
    #
    # Enchaînées, c'est le montage des vidéos de 2025 : les images se
    # succèdent, chacune se défait pour découvrir la suivante, et ce que
    # l'une abandonne, l'autre l'occupe entièrement. La couverture vaut
    # toujours 1 : il n'y a jamais de fond. Voir chain.py.
    chained: bool = False

    # Le décalage entre couches, et ce qui résiste jusqu'au bout
    stagger: float = 0.45
    residue: float = 0.12

    # De combien les matières choisies s'intensifient au bout de l'axe, et
    # quelle part y prend la déchirure héritée du datamoshing. À tearing 0,
    # le temps ne fait plus que mûrir ce qu'on a posé, et évider.
    ripening: float = 1.0
    tearing: float = 1.0

    # Le tri de pixels
    min_segment: float = 0.03
    max_segment: float = 0.10
    segment_growth: float = 1.5   # les segments s'allongent en pourrissant
    channel_shift: float = 0.015

    # Les quatre bascules héritées de l'original. Voir video.py pour ce que
    # chacune reproduit, et pourquoi trois d'entre elles sont des corrections.
    enhanced: bool = True
    channel_quirk: bool = True
    sort_visible: bool = True
    even_dissolve: bool = True

    def at(self, index: int) -> float:
        """L'avancement global, de 0 à 1."""
        return min(1.0, max(0.0, index / max(1, self.frames - 1)))


def ripen(effect: Effect, amount: float) -> Effect:
    """Fait mûrir une matière : c'est elle que le temps travaille d'abord.

    Chaque matière mûrit selon sa nature. Un grain ou une longueur grossit.
    Le bitmap ne peut pas dépasser 1 : il durcit vers le noir et blanc
    francs. La saturation part de 1, donc c'est son écart à 1 qui grandit,
    sans quoi la multiplier la ferait bondir dès le premier pas.
    """
    if amount <= 0:
        return effect
    if effect.name == "bitmap":
        monte = effect.strength + (1.0 - effect.strength) * min(1.0, amount)
        return replace(effect, strength=float(min(1.0, monte)))
    if effect.name == "saturate":
        return replace(effect, strength=1.0 + (effect.strength - 1.0) * (1.0 + amount))
    return replace(effect, strength=effect.strength * (1.0 + amount))


def layer_progress(decay: Decay, moment: float, layer: int, layers: int) -> float:
    """L'avancement propre à une couche, à un instant donné de l'axe.

    La dernière image choisie est la plus exposée : elle part la première.
    """
    if layers < 2 or decay.stagger <= 0:
        return moment

    rank = layers - 1 - layer          # 0 pour la couche du dessus
    start = decay.stagger * rank / layers
    span = max(1e-6, 1.0 - decay.stagger * (layers - 1) / layers)
    return float(np.clip((moment - start) / span, 0.0, 1.0))


def decompose_layer(
    image: np.ndarray, keep: np.ndarray, progress: float, side: int,
    decay: Decay, rng: random.Random, flips,
) -> np.ndarray:
    """Défait une couche : ce qu'on voit encore est ce qui se trie.

    L'original triait la zone en train de disparaître, si bien que son tri ne
    se voyait jamais ; `sort_visible` garde cette possibilité pour retrouver
    son rendu exact.
    """
    if progress <= 0 or decay.tearing <= 0:
        return image

    # La déchirure se dose : à part réduite elle mord moins loin et décale
    # moins, plutôt que de s'arrêter net.
    progress = progress * float(np.clip(decay.tearing, 0.0, 1.0))
    grow = 1.0 + decay.segment_growth * progress
    minimum = max(2, int(decay.min_segment * side * grow))
    maximum = max(minimum + 1, int(decay.max_segment * side * grow))
    vertical = rng.random() < 0.5
    zone = keep > 0.5 if decay.sort_visible else keep < 0.5

    if decay.enhanced:
        undone = sort_channels_separately(
            image, zone, vertical, side, rng, flips, decay.channel_quirk,
            minimum, maximum,
        )
    else:
        undone = pixel_sort(
            image, zone, rng.choice(("brightness", "hue", "saturation")),
            vertical, minimum, maximum, flips,
        )
    return shift_channels(undone, int(decay.channel_shift * side * progress), rng)


def compose_at(
    recipe: Recipe, decay: Decay, size: int, index: int,
    sources=None, fields=None, avance=None,
) -> np.ndarray:
    """La composition à un instant de l'axe.

    Chaque frame se calcule seule, sans rejouer les précédentes : c'est ce qui
    permet de se déplacer librement dans le temps, et de prélever un instant
    en pleine définition sans capture d'écran.

    `fields` évite de recalculer les cartes de saillance à chaque frame d'une
    séquence ; `prepare_fields` les produit une fois pour toutes. `avance`,
    appelée après chaque couche, permet de suivre un tirage qui dure : en
    grand format, une couche demande plusieurs secondes.
    """
    if decay.chained:
        from atelier.chain import frame_at

        return frame_at(recipe, decay, size, index, sources, avance=avance)

    images = sources if sources is not None else [load_source(p) for p in recipe.sources]
    if not images:
        raise ValueError("aucune image source")

    moment = decay.at(index)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    weights = np.zeros((size, size), dtype=np.float32)

    for layer, (image, placement) in enumerate(zip(images, placements_for(recipe))):
        target, x, y = layer_geometry(image, placement, size, recipe.full_frame)
        resized = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
        card = (fields[layer] if fields is not None else saliency_of(image))
        saliency = smooth_mask(
            cv2.resize(card, target, interpolation=cv2.INTER_LINEAR),
            recipe.smoothness, recipe.edge_blur,
        )

        progress = layer_progress(decay, moment, layer, len(images))

        # Les matières posées mûrissent le long de l'axe : c'est ce qu'on a
        # choisi qui se décompose, et non une dégradation venue par-dessus.
        matieres = effects_for(recipe, layer)
        if progress > 0 and decay.ripening:
            matieres = tuple(ripen(e, progress * decay.ripening) for e in matieres)

        treated = apply_effects(resized, saliency, recipe, size, matieres)

        if progress > 0:
            field, even = dissolve_field(saliency, decay.even_dissolve)
            # La bavure vaut aussi pour l'évidement : c'est lui qui découpe
            # la matière au rasoir sur le fond, et ces bords francs sont ce
            # qu'on voit le plus à mi-axe.
            keep = retention_at(
                field, progress * (1.0 - decay.residue), even,
                softness=float(recipe.bleed) * BLEED_SOFTNESS * size,
            )
            rng, flips = frame_draw(recipe.seed, layer, index)
            treated = decompose_layer(
                treated, keep, progress, size, decay, rng, flips
            )
        else:
            keep = np.ones(saliency.shape, dtype=np.float32)

        weight = saliency.astype(np.float32) / 255.0 * keep
        blend_into(canvas, weights, treated, weight, x, y)
        if avance is not None:
            avance(layer + 1, len(images))

    covered = weights > 1e-6
    normaliser(canvas, weights, recipe.fade)

    # La normalisation rend sa pleine intensité au moindre pixel couvert :
    # adoucir les masques en amont ne servirait à rien, le bord de la
    # matière redeviendrait une coupure. C'est donc ici, sur la présence
    # elle-même, que la bavure doit agir — la matière s'éteint alors vers le
    # fond au lieu de s'y arrêter net.
    # Seulement une fois l'axe engagé : à l'instant 0 il n'y a pas encore
    # d'évidement, et la composition doit rester celle que rend `render`.
    fondu = float(recipe.bleed) * BLEED_SOFTNESS * size if moment > 0 else 0.0
    if fondu > 0.5:
        # Un flou de ce rayon coûte cher en pleine résolution, et n'y gagne
        # rien : on le calcule sur une réduction et on le rétablit. À sigma
        # égal, l'écart ne se voit pas, et l'aperçu y gagne un quart de
        # seconde à chaque mouvement de curseur.
        petit = max(64, size // 4)
        reduit = cv2.resize(covered.astype(np.float32), (petit, petit),
                            interpolation=cv2.INTER_AREA)
        reduit = cv2.GaussianBlur(reduit, (0, 0), fondu * petit / size)
        presence = cv2.resize(reduit, (size, size), interpolation=cv2.INTER_LINEAR)
        canvas *= np.clip(presence * 1.1, 0.0, 1.0)[:, :, None]

    return np.clip(canvas, 0, 255).astype(np.uint8)


def prepare_fields(recipe: Recipe, sources=None) -> list[np.ndarray]:
    """Les cartes de saillance, calculées une fois pour toute la séquence."""
    images = sources if sources is not None else [load_source(p) for p in recipe.sources]
    return [saliency_of(image) for image in images]


def sequence(recipe: Recipe, decay: Decay, size: int, sources=None):
    """Tout l'axe, une frame après l'autre."""
    images = sources if sources is not None else [load_source(p) for p in recipe.sources]
    if not images:
        raise ValueError("aucune image source")

    if decay.chained:
        from atelier.chain import sequence as enchainer

        yield from enchainer(recipe, decay, size, images)
        return

    fields = prepare_fields(recipe, images)
    for index in range(count_frames(decay, len(images))):
        yield compose_at(recipe, decay, size, index, images, fields)


def count_frames(decay: Decay, images: int = 2) -> int:
    """La longueur de l'axe.

    Superposées, elle ne dépend que du réglage : une image de plus allonge la
    pile, pas la durée — c'est une couche de compost supplémentaire, pas une
    diapositive de plus.

    Enchaînées, l'axe garde la même longueur mais se partage entre les
    transitions, au nombre entier d'images près. Ajouter une image resserre
    donc les enchaînements au lieu d'allonger la vidéo dans le dos.
    """
    if decay.chained:
        from atelier.chain import compte

        return compte(decay, images)
    return max(2, decay.frames)
