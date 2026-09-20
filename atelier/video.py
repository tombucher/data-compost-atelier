"""Voie vidéo : une transition qui décompose une image pour en révéler une autre.

Reprise de `datamoshing4.py`, qui a produit les vidéos de mars 2025 et, par
capture d'écran, les 116 images de `export-vid_sauv/SCREENSHOT`.

Trois changements, les mêmes que pour la voie image.

**Le hasard est retenu.** La méthode de tri et la direction du décalage RVB
étaient tirées par le `random` global, sans graine : une séquence ne pouvait
pas être rejouée. Elles viennent maintenant du tirage de la recette.

**Les longueurs suivent la taille.** Segments et décalages étaient exprimés en
pixels, calibrés pour du 1024 : à 3000 px l'effet devenait quatre fois plus
fin. Ils sont désormais des fractions de l'image.

Et surtout : une image extraite d'ici sort à la résolution du rendu, pas à
celle de l'écran. La capture manuelle coûtait un quart de la définition —
2254 px saisis sur des vidéos qui en contenaient 3016.

Le tri est aussi vectorisé, mais le gain est modeste : ×1,8 en 3000 px, soit
5 minutes au lieu de 10 pour une séquence de 150 frames. Le travail lourd
était déjà dans numpy chez l'original ; seule la détection des segments était
en Python. C'est surtout la lisibilité qui y gagne.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np

from atelier.engine import Recipe, SOURCE_SPAN, saliency_of

SORT_METHODS = ("brightness", "hue", "saturation")


@dataclass(frozen=True)
class Motion:
    """Ce qui ne concerne que la vidéo.

    Les longueurs sont en fraction du côté de l'image : `min_segment` à 0.03
    vaut 30 px sur 1024 et 90 px sur 3000, ce qui donne le même geste.
    """

    frames_per_transition: int = 24
    fps: int = 24
    min_segment: float = 0.03     # 30 px sur 1024, comme l'original au départ
    max_segment: float = 0.10     # 100 px sur 1024
    segment_growth: float = 1.5   # les segments s'allongent au fil de la transition
    channel_shift: float = 0.015  # 15 px sur 1024
    enhanced: bool = True         # canaux traités séparément, comme l'original
    # Le tirage de méthodes de l'original, qui laisse le rouge intact.
    # Voir sort_channels_separately(). À décocher pour trier les trois canaux.
    channel_quirk: bool = True
    # Trier l'image là où elle est visible, et non là où elle disparaît.
    # L'original faisait l'inverse et son tri restait invisible ; voir
    # transition(). Mettre False pour retrouver son comportement exact.
    sort_visible: bool = True
    # Répartir la dissolution sur les quantiles de la saillance plutôt que sur
    # ses valeurs brutes. Voir mask_sequence(). False retrouve l'original.
    even_dissolve: bool = True


def sort_key(image: np.ndarray, method: str) -> np.ndarray:
    """Valeur de tri par pixel, dans [0, 1023]."""
    if method == "brightness":
        return image.sum(axis=2).astype(np.int32)  # 0..765
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channel = 0 if method == "hue" else 1
    return hsv[:, :, channel].astype(np.int32)


def pixel_sort(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "brightness",
    vertical: bool = False,
    min_length: int = 20,
    max_length: int = 100,
    flips: np.random.Generator | None = None,
) -> np.ndarray:
    """Trie les pixels par segments continus, à l'intérieur du masque.

    Vectorisé par une astuce de groupes : on numérote les pixels de sorte que
    le numéro soit constant sur un segment à trier et unique ailleurs, puis on
    trie une seule fois sur la clé `numéro × 1024 + valeur`. Les pixels hors
    segment, seuls dans leur groupe, ne bougent pas ; ceux d'un même segment
    se réordonnent entre eux sans jamais en sortir.

    `flips` reproduit l'inversion que l'original tirait une fois sur deux :
    certains segments courent du sombre au clair, d'autres l'inverse. C'était
    du `random` global, donc perdu ; c'est maintenant tiré de la graine.

    L'original faisait le même travail avec deux boucles Python imbriquées.
    """
    if min_length < 2:
        return image

    work = image if vertical else np.transpose(image, (1, 0, 2))
    flags = mask if vertical else mask.T
    flags = np.asarray(flags, dtype=bool)
    height = work.shape[0]

    # Numéro d'ordre à l'intérieur de chaque plage continue du masque
    starts = flags & ~np.vstack([np.zeros((1, flags.shape[1]), bool), flags[:-1]])
    run_id = np.cumsum(starts, axis=0)
    position = np.arange(height)[:, None]
    run_start = np.where(starts, position, 0)
    run_start = np.maximum.accumulate(run_start, axis=0)
    within = position - run_start

    # Une plage trop longue est découpée, comme dans l'original
    piece = within // max_length
    piece_id = run_id * (height // max_length + 2) + piece

    # Une plage trop courte n'est pas triée
    sortable = flags.copy()
    for column in range(flags.shape[1]):
        ids = piece_id[:, column]
        keep = flags[:, column]
        if not keep.any():
            continue
        counts = np.bincount(ids[keep])
        too_short = counts < min_length
        sortable[:, column] = keep & ~too_short[np.clip(ids, 0, counts.size - 1)]

    # Groupes : constants sur un segment triable, uniques ailleurs
    boundary = ~sortable | (piece_id != np.vstack(
        [np.full((1, flags.shape[1]), -1), piece_id[:-1]]
    )) | ~np.vstack([np.zeros((1, flags.shape[1]), bool), sortable[:-1]])
    group = np.cumsum(boundary, axis=0)

    values = sort_key(work, method)
    if flips is not None:
        # Un sens de tri par segment, décidé une fois pour toutes
        draw = flips.random((int(group.max()) + 1, group.shape[1])) < 0.5
        columns = np.arange(group.shape[1])[None, :]
        reversed_here = draw[group, columns]
        values = np.where(reversed_here, 1023 - values, values)

    order = np.argsort(group.astype(np.int64) * 1024 + values, axis=0, kind="stable")
    sorted_work = np.take_along_axis(work, order[:, :, None], axis=0)

    return sorted_work if vertical else np.transpose(sorted_work, (1, 0, 2))


def sort_channels_separately(
    image: np.ndarray,
    mask: np.ndarray,
    vertical: bool,
    side: int,
    rng: random.Random,
    flips: np.random.Generator | None = None,
    quirk: bool = True,
) -> np.ndarray:
    """Trie chaque canal séparément : c'est de là que vient la séparation colorée.

    Chaque canal reçoit sa propre longueur de segment, sa propre méthode, et
    le vert part dans la direction opposée. L'original tirait tout cela au
    hasard global ; ici tout vient de la graine.

    Une bizarrerie de l'original est conservée, parce qu'elle fait l'image.
    Chaque canal est répliqué en gris avant d'être trié — or trier un gris par
    teinte ou par saturation ne change rien, ces valeurs y étant toutes
    nulles. Comme le rouge ne tirait qu'entre « saturation » et « teinte », il
    n'était **jamais** trié, et le vert et le bleu une fois sur deux. C'est
    cette asymétrie qui sépare les couleurs. `quirk=False` trie les trois.
    """
    blue, green, red = cv2.split(image)

    # Longueurs relatives à la taille, comme partout ailleurs. Les valeurs de
    # l'original (20-50, 30-80, 40-100) étaient calibrées pour du 1024.
    def span(low, high, ceiling):
        lo = max(2, int(rng.randint(low, high) * side / 1024))
        return lo, max(lo + 1, int(ceiling * side / 1024))

    blue_min, blue_max = span(20, 50, 150)
    green_min, green_max = span(30, 80, 180)
    red_min, red_max = span(40, 100, 200)

    if quirk:
        blue_method = rng.choice(("brightness", "hue"))
        green_method = rng.choice(("brightness", "saturation"))
        red_method = rng.choice(("saturation", "hue"))
    else:
        blue_method = green_method = red_method = "brightness"

    def one(channel, method, vert, low, high):
        grey = cv2.merge([channel, channel, channel])
        return pixel_sort(grey, mask, method, vert, low, high, flips)[:, :, 0]

    return cv2.merge([
        one(blue, blue_method, vertical, blue_min, blue_max),
        one(green, green_method, not vertical, green_min, green_max),
        one(red, red_method, vertical, red_min, red_max),
    ])


def shift_channels(image: np.ndarray, shift: int, rng: random.Random) -> np.ndarray:
    """Décale les canaux rouge et bleu en sens opposés.

    La direction était tirée par le `random` global de l'original ; elle vient
    maintenant du tirage de la recette, pour qu'une séquence se rejoue.
    """
    if shift <= 0:
        return image
    blue, green, red = cv2.split(image)
    if rng.random() < 0.7:  # horizontal sept fois sur dix, comme l'original
        red = np.roll(red, shift, axis=1)
        blue = np.roll(blue, -shift, axis=1)
    else:
        red = np.roll(red, shift // 2, axis=0)
        blue = np.roll(blue, -shift // 2, axis=0)
    return cv2.merge([blue, green, red])


def mask_sequence(saliency: np.ndarray, frames: int,
                  even: bool = True) -> list[np.ndarray]:
    """Seuils croissants sur la saillance : l'image se retire par degrés.

    Le résidu spectral produit une carte très concentrée — sur les
    photographies du projet, la médiane est à 0,02 et la moyenne à 0,04.
    Répartir les seuils de 0 à 1 comme le faisait l'original vidait donc
    l'image en deux images : 100 % conservé, puis 16 %, puis 6 %, et plus
    rien ne bougeait pendant les vingt suivantes. C'est pourquoi aucune
    décomposition n'était visible.

    Les seuils suivent maintenant les quantiles de la carte : chaque image
    retire une part comparable, et la dissolution occupe toute la séquence.
    `even=False` retrouve le comportement d'origine.
    """
    field = saliency.astype(np.float32)
    if field.max() > 1.0:
        field = field / 255.0
    field = cv2.GaussianBlur(field, (15, 15), 0)

    # Une carte sans relief — image unie, scan de papier blanc — rendrait
    # tous les quantiles égaux, et l'image ne se dissoudrait jamais : la
    # seconde n'arriverait pas. On se rabat alors sur un balayage spatial,
    # qui donne au moins une transition.
    if float(field.max() - field.min()) < 1e-3:
        width = field.shape[1]
        field = np.tile(np.linspace(1.0, 0.0, width, dtype=np.float32), (field.shape[0], 1))
        even = False

    # Exposant 0.8 : on s'attarde sur le milieu de la dissolution, là où se
    # joue la décomposition.
    steps = np.power(np.linspace(0.0, 1.0, max(frames, 2)), 0.8)
    thresholds = np.quantile(field, steps) if even else steps
    # Le dernier masque doit être vide, pour que la seconde image arrive
    # entièrement quelle que soit la distribution.
    thresholds = np.asarray(thresholds, dtype=np.float64)
    thresholds[-1] = float(field.max()) + 1e-6
    return [(field >= t).astype(np.float32) for t in thresholds]


def transition_masks(first, second, frames, even: bool = True):
    """Les deux séquences de masques d'une transition.

    La première image se retire par seuils croissants ; la seconde reste
    absente le premier tiers puis se révèle de la même manière.
    """
    blur = (31, 31)
    masks_first = mask_sequence(
        cv2.GaussianBlur(saliency_of(first), blur, 0), frames, even
    )
    late = max(2 * frames // 3, 1)
    masks_second = [np.zeros(second.shape[:2], np.float32)] * (frames - late)
    masks_second += mask_sequence(
        cv2.GaussianBlur(saliency_of(second), blur, 0), late, even
    )
    return masks_first, masks_second[:frames], late


def frame_draw(seed: int, pair: int, index: int):
    """Le tirage propre à une frame.

    Dérivé de (graine, transition, numéro de frame) plutôt que d'un tirage
    qui avance à chaque image : toute frame se calcule alors seule, ce qui
    permet de naviguer dans une séquence sans la rejouer depuis le début.
    """
    # Une chaîne plutôt qu'un tuple : Python 3.14 n'accepte plus les tuples
    # comme graine de random.Random.
    return (
        random.Random(f"{seed}:{pair}:{index}"),
        np.random.default_rng([int(seed), int(pair), int(index)]),
    )


def transition(
    first: np.ndarray,
    second: np.ndarray,
    motion: Motion,
    recipe: Recipe,
    pair: int = 0,
):
    """Engendre les frames d'une transition entre deux images de même taille."""
    frames = max(2, motion.frames_per_transition)
    side = max(first.shape[:2])
    masks_first, masks_second, late = transition_masks(
        first, second, frames, motion.even_dissolve
    )

    for index in range(frames):
        yield _one_frame(first, second, masks_first[index], masks_second[index],
                         index, frames, late, side, motion, recipe.seed, pair,
                         recipe)


def _one_frame(first, second, keep, reveal, index, frames, late, side,
               motion, seed, pair, recipe=None):
    """Une frame, calculable indépendamment des autres.

    Les matières de la voie image — trame, bitmap, pixellisation, saturation —
    s'appliquent aussi ici, chacune avec le réglage propre de son image. Rien
    n'obligeait à séparer les deux voies.
    """
    if True:
        rng, flips = frame_draw(seed, pair, index)
        progress = index / (frames - 1)

        # Les segments s'allongent à mesure que l'image se défait
        grow = 1.0 + motion.segment_growth * progress
        minimum = max(2, int(motion.min_segment * side * grow))
        maximum = max(minimum + 1, int(motion.max_segment * side * grow))

        vertical = index < frames / 2
        method = rng.choice(SORT_METHODS)

        # L'original triait l'image 1 sur `keep < 0.5`, puis la composait
        # avec le poids `keep` : les deux zones étant complémentaires, le tri
        # tombait exactement là où l'image n'est pas affichée et ne se voyait
        # jamais. Son propre commentaire annonçait pourtant une image « qui
        # reste visible et se dégrade ». On trie donc ce qu'on montre.
        zone_first = keep > 0.5 if motion.sort_visible else keep < 0.5
        if motion.enhanced:
            undone = sort_channels_separately(
                first, zone_first, vertical, side, rng, flips, motion.channel_quirk
            )
        else:
            undone = pixel_sort(
                first, zone_first, method, vertical, minimum, maximum, flips
            )
        shift = int(motion.channel_shift * side * progress)
        undone = shift_channels(undone, shift, rng)

        if index >= frames - late:
            # L'image 2 n'est montrée que là où l'image 1 s'est retirée :
            # c'est cette part-là qu'il faut travailler.
            zone_second = (reveal > 0.5) & (keep < 0.5) if motion.sort_visible \
                else reveal > 0.5
            if motion.enhanced:
                revealed = sort_channels_separately(
                    second, zone_second, not vertical, side, rng, flips,
                    motion.channel_quirk,
                )
            else:
                revealed = pixel_sort(
                    second, zone_second, method, not vertical,
                    max(2, minimum // 2), max(3, maximum // 2), flips,
                )
        else:
            revealed = second

        blend = keep[:, :, None]
        composed = (revealed * (1.0 - blend) + undone * blend).astype(np.uint8)

        if recipe is not None:
            from atelier.engine import apply_effects, effects_for, saliency_of

            matieres = effects_for(recipe, pair)
            if matieres:
                composed = apply_effects(
                    composed, saliency_of(composed), recipe, side, matieres
                )
        return composed


def render_sequence(recipe: Recipe, motion: Motion, size: int, sources=None):
    """Enchaîne les transitions entre images successives.

    Les images sont cadrées comme dans la voie image, pour qu'une vidéo et une
    composition nées de la même recette restent parentes.
    """
    framed = frame_canvases(recipe, size, sources)
    if len(framed) < 2:
        raise ValueError("il faut au moins deux images pour une transition")

    for pair, (first, second) in enumerate(zip(framed, framed[1:])):
        yield from transition(first, second, motion, recipe, pair)


def frame_canvases(recipe: Recipe, size: int, sources=None) -> list[np.ndarray]:
    """Les images sources, cadrées au carré comme dans la voie image."""
    from atelier.engine import load_source

    images = sources if sources is not None else [load_source(p) for p in recipe.sources]
    # La toile fait exactement la taille demandée ; l'image y occupe la même
    # part que dans la voie image, centrée. Demander 4000 doit donner 4000.
    span = max(1, int(round(SOURCE_SPAN * size)))
    framed = []
    for image in images:
        height, width = image.shape[:2]
        scale = span / max(height, width)
        resized = cv2.resize(
            image, (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.zeros((size, size, 3), np.uint8)
        y = (size - resized.shape[0]) // 2
        x = (size - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        framed.append(canvas)
    return framed


def render_frame(recipe: Recipe, motion: Motion, size: int, index: int,
                 sources=None) -> np.ndarray:
    """Rend une seule frame de la séquence, sans calculer les précédentes.

    C'est ce qui remplace la capture d'écran : on parcourt la séquence, on
    s'arrête sur l'instant voulu, et on l'exporte à la taille du tirage. La
    capture manuelle plafonnait à la définition de l'écran.
    """
    framed = frame_canvases(recipe, size, sources)
    if len(framed) < 2:
        raise ValueError("il faut au moins deux images pour une transition")

    per_pair = max(2, motion.frames_per_transition)
    total = (len(framed) - 1) * per_pair
    index = max(0, min(int(index), total - 1))
    pair, local = divmod(index, per_pair)

    first, second = framed[pair], framed[pair + 1]
    masks_first, masks_second, late = transition_masks(
        first, second, per_pair, motion.even_dissolve
    )
    return _one_frame(
        first, second, masks_first[local], masks_second[local],
        local, per_pair, late, max(first.shape[:2]), motion, recipe.seed, pair,
        recipe,
    )


def count_frames(recipe: Recipe, motion: Motion) -> int:
    return max(0, len(recipe.sources) - 1) * max(2, motion.frames_per_transition)


def write_video(frames, path, fps: int = 24) -> int:
    """Écrit les frames en H.264, ou en mp4v si le codec manque.

    Renvoie le nombre de frames écrites. Le flux est consommé au fil de l'eau :
    une séquence en 3000 px ne tient pas en mémoire.
    """
    writer = None
    written = 0
    try:
        for frame in frames:
            if writer is None:
                height, width = frame.shape[:2]
                for code in ("avc1", "mp4v"):
                    candidate = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*code), fps, (width, height)
                    )
                    if candidate.isOpened():
                        writer = candidate
                        break
                    candidate.release()
                if writer is None:
                    raise RuntimeError("aucun encodeur vidéo disponible")
            writer.write(frame)
            written += 1
    finally:
        if writer is not None:
            writer.release()
    return written
