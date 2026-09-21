"""L'enchaînement : une image se défait pour en découvrir une autre.

C'est le montage des vidéos de février 2025 — `resultat.mp4` et les siennes —,
repris de `datamoshing4.py`. Il ne compose rien : à chaque instant il n'y a
qu'une image, ou deux en train de se relayer. C'est ce qui explique qu'on n'y
voie jamais de fond. La ligne qui fait tout, chez l'original :

    combined = processed2 * (1.0 - mask1) + processed1 * mask1

Les deux parts sont complémentaires : ce que la première abandonne, la seconde
l'occupe entièrement. La couverture vaut donc exactement 1 partout, à tout
instant, et il n'y a pas de noir possible.

## Pourquoi c'est un mode et non un réglage

La pile et l'enchaînement ne sont pas deux dosages d'une même chose. Dans la
pile, toutes les images sont là en même temps, pondérées par leur saillance,
et le temps défait l'ensemble : ce que l'évidement creuse finit par déboucher
sur le fond, puisqu'il n'y a plus rien dessous. Dans l'enchaînement, les
images se succèdent, et ce que l'une abandonne, la suivante le prend. L'une
composte, l'autre monte.

`stagger`, `residue` et la disparition n'ont donc pas de sens ici : il n'y a
pas de couches à décaler, et rien à faire survivre — une transition doit
arriver au bout, sinon l'image suivante n'apparaîtrait jamais entière. Le
reste de l'axe s'applique : le mûrissement fait grossir les matières qu'on a
posées, la déchirure dose le tri et le décalage RVB.
"""

from __future__ import annotations

import cv2
import numpy as np

from atelier.engine import (
    Recipe,
    apply_effects,
    effects_for,
    layer_geometry,
    load_source,
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

# Flou appliqué à la carte avant d'en tirer la dissolution. Celui de
# l'original, qui donne à la transition ses grandes plages molles plutôt
# qu'un grignotage pixel par pixel.
DISSOLVE_BLUR = (31, 31)

# Part de la transition pendant laquelle la seconde image se trie à son tour.
# L'original la faisait « apparaître plus tôt » sur les deux derniers tiers.
LATE = 2 / 3


def poser(canvas: np.ndarray, tile: np.ndarray, x: int, y: int) -> None:
    """Colle une image sur la toile, en coupant ce qui en sort."""
    hauteur, largeur = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1 = min(largeur, x + tile.shape[1])
    y1 = min(hauteur, y + tile.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    canvas[y0:y1, x0:x1] = tile[y0 - y:y1 - y, x0 - x:x1 - x]


def prepare(recipe: Recipe, size: int, sources=None):
    """Chaque image posée seule sur sa toile, avec ses deux cartes.

    Le cadrage est celui de la pile — même part de toile, même placement —
    pour qu'une vidéo et une composition nées de la même recette restent
    parentes. En plein cadre, l'image couvre tout et il ne reste aucun fond,
    ce qui est le cadrage des vidéos d'origine.
    """
    images = sources if sources is not None else [load_source(p) for p in recipe.sources]
    toiles, masques, champs = [], [], []

    for image, placement in zip(images, placements_for(recipe)):
        target, x, y = layer_geometry(image, placement, size, recipe.full_frame)
        canvas = np.zeros((size, size, 3), np.uint8)
        poser(canvas, cv2.resize(image, target, interpolation=cv2.INTER_AREA), x, y)
        carte = saliency_of(canvas)

        toiles.append(canvas)
        # Deux usages, deux préparations : les matières mordent sur la carte
        # réglée par le lissage et les contours, la dissolution sur la carte
        # très floutée de l'original.
        masques.append(smooth_mask(carte, recipe.smoothness, recipe.edge_blur))
        champs.append(dissolve_field(cv2.GaussianBlur(carte, DISSOLVE_BLUR, 0)))

    return toiles, masques, champs


def par_paire(decay, images: int) -> int:
    """Combien d'images pour une transition.

    L'axe garde la longueur qu'on lui a donnée, et se partage entre les
    enchaînements : la timeline reste au même compte que la vidéo écrite, et
    ajouter une image ne double pas la durée dans le dos.
    """
    return max(2, int(decay.frames) // max(1, images - 1))


def compte(decay, images: int) -> int:
    """La longueur réelle de l'axe, une fois le partage fait."""
    return par_paire(decay, images) * max(1, images - 1)


def matieres_de(recipe: Recipe, rang: int, decay, progress: float):
    """Les matières d'une image, mûries à l'avancement de la transition."""
    from atelier.decay import ripen

    posees = effects_for(recipe, rang)
    if progress <= 0 or not decay.ripening:
        return posees
    return tuple(ripen(e, progress * decay.ripening) for e in posees)


def travailler(image, masque, recipe, size, rang, decay, progress):
    """Pose les matières choisies sur une image, avant qu'elle se défasse."""
    matieres = matieres_de(recipe, rang, decay, progress)
    if not matieres:
        return image
    return apply_effects(image, masque, recipe, size, matieres)


def frame_at(recipe: Recipe, decay, size: int, index: int,
             sources=None, prepared=None, avance=None) -> np.ndarray:
    """Une image de l'enchaînement, calculable seule.

    Comme pour la pile, aucune frame ne dépend des précédentes : on peut se
    déplacer librement sur l'axe, et prélever un instant en pleine définition
    sans passer par une capture d'écran.
    """
    toiles, masques, champs = prepared if prepared is not None \
        else prepare(recipe, size, sources)
    if len(toiles) < 2:
        raise ValueError("il faut au moins deux images pour un enchaînement")

    longueur = par_paire(decay, len(toiles))
    index = max(0, min(int(index), compte(decay, len(toiles)) - 1))
    paire, local = divmod(index, longueur)
    paire = min(paire, len(toiles) - 2)

    progress = local / (longueur - 1)
    tard = max(2, int(longueur * LATE))
    rng, flips = frame_draw(recipe.seed, paire, local)
    cote = size
    morsure = float(np.clip(decay.tearing, 0.0, 1.0))

    # Ce qu'il reste de la première image. Elle doit arriver à rien : sans
    # cela, la suivante n'apparaîtrait jamais entière et l'enchaînement
    # n'enchaînerait pas. C'est pourquoi `residue` ne s'applique pas ici.
    champ, egal = champs[paire]
    keep = retention_at(champ, progress, egal)

    grandir = 1.0 + decay.segment_growth * progress * morsure
    mini = max(2, int(decay.min_segment * cote * grandir))
    maxi = max(mini + 1, int(decay.max_segment * cote * grandir))
    vertical = local < longueur / 2

    premiere = travailler(toiles[paire], masques[paire], recipe, size,
                          paire, decay, progress)
    if avance is not None:
        avance(1, 2)

    # L'original triait la première image là où elle *n'est pas* montrée, si
    # bien que son tri ne se voyait jamais ; `sort_visible` garde cette
    # possibilité pour retrouver son rendu exact.
    #
    # Rien au premier instant d'une transition, comme au premier instant de
    # la pile : on doit voir l'image telle qu'on l'a réglée avant de la voir
    # se défaire. Cela raccorde aussi les transitions entre elles, puisque la
    # précédente finit sur cette même image intacte.
    if morsure > 0 and progress > 0:
        zone = keep > 0.5 if decay.sort_visible else keep < 0.5
        if decay.enhanced:
            premiere = sort_channels_separately(
                premiere, zone, vertical, cote, rng, flips,
                decay.channel_quirk, mini, maxi,
            )
        else:
            premiere = pixel_sort(
                premiere, zone, rng.choice(("brightness", "hue", "saturation")),
                vertical, mini, maxi, flips,
            )
        premiere = shift_channels(
            premiere, int(decay.channel_shift * cote * progress * morsure), rng
        )

    seconde = travailler(toiles[paire + 1], masques[paire + 1], recipe, size,
                         paire + 1, decay, progress)
    if avance is not None:
        avance(2, 2)

    # La seconde image n'est travaillée que sur la part déjà libérée : ailleurs
    # elle n'est pas montrée, et la trier coûterait pour rien.
    if morsure > 0 and local >= longueur - tard:
        venue = retention_at(
            champs[paire + 1][0],
            (local - (longueur - tard)) / max(1, tard - 1),
            champs[paire + 1][1],
        )
        zone = (venue > 0.5) & (keep < 0.5) if decay.sort_visible else venue > 0.5
        if decay.enhanced:
            seconde = sort_channels_separately(
                seconde, zone, not vertical, cote, rng, flips,
                decay.channel_quirk, mini, maxi,
            )
        else:
            seconde = pixel_sort(
                seconde, zone, rng.choice(("brightness", "hue", "saturation")),
                not vertical, max(2, mini // 2), max(3, maxi // 2), flips,
            )

    melange = keep[:, :, None]
    return (seconde * (1.0 - melange) + premiere * melange) \
        .clip(0, 255).astype(np.uint8)


def sequence(recipe: Recipe, decay, size: int, sources=None):
    """Tout l'enchaînement, une image après l'autre."""
    prepared = prepare(recipe, size, sources)
    for index in range(compte(decay, len(prepared[0]))):
        yield frame_at(recipe, decay, size, index, prepared=prepared)
