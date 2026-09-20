"""Les matières du datamoshing : trier, déchirer, désaligner les canaux.

Reprise de `datamoshing4.py`, qui a produit les vidéos de mars 2025 et, par
capture d'écran, les 116 images de `export-vid_sauv/SCREENSHOT`. Ce module
n'en garde que les gestes ; le temps qui les enchaîne est dans `decay.py`,
et ils servent aussi bien à une image fixe.

Trois changements, les mêmes que pour la composition.

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
import cv2
import numpy as np

SORT_METHODS = ("brightness", "hue", "saturation")


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
    minimum: int | None = None,
    maximum: int | None = None,
    cross: bool = True,
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

    Sans `minimum`/`maximum`, les longueurs de l'original s'appliquent. Avec,
    elles se règlent — mais les rapports entre canaux sont conservés, car
    c'est leur écart qui fait la séparation, pas leur valeur absolue.

    `cross` envoie le vert perpendiculairement aux deux autres, comme le
    faisait l'original. Cela n'y était presque jamais visible : la
    bizarrerie ci-dessus laissait le rouge intact et le bleu trié une fois
    sur deux, si bien que le vert était souvent le seul canal réellement
    trié, et l'on ne voyait qu'une direction. Dès que les trois canaux
    travaillent, la croix saute aux yeux — d'où le réglage.
    """
    blue, green, red = cv2.split(image)

    # Longueurs relatives à la taille, comme partout ailleurs. Les valeurs de
    # l'original (20-50, 30-80, 40-100) étaient calibrées pour du 1024 ; les
    # rapports 0,7 / 1 / 1,4 en gardent l'écart quand on règle soi-même.
    def span(low, high, ceiling, ratio):
        if minimum is None:
            lo = max(2, int(rng.randint(low, high) * side / 1024))
            return lo, max(lo + 1, int(ceiling * side / 1024))
        rng.random()   # même nombre de tirages dans les deux branches
        lo = max(2, int(minimum * ratio))
        return lo, max(lo + 1, int((maximum or minimum * 3) * ratio))

    blue_min, blue_max = span(20, 50, 150, 0.7)
    green_min, green_max = span(30, 80, 180, 1.0)
    red_min, red_max = span(40, 100, 200, 1.4)

    if quirk:
        blue_method = rng.choice(("brightness", "hue"))
        green_method = rng.choice(("brightness", "saturation"))
        red_method = rng.choice(("saturation", "hue"))
    else:
        blue_method = green_method = red_method = "brightness"

    def one(channel, method, vert, low, high):
        grey = cv2.merge([channel, channel, channel])
        return pixel_sort(grey, mask, method, vert, low, high, flips)[:, :, 0]

    vert_vertical = (not vertical) if cross else vertical
    return cv2.merge([
        one(blue, blue_method, vertical, blue_min, blue_max),
        one(green, green_method, vert_vertical, green_min, green_max),
        one(red, red_method, vertical, red_min, red_max),
    ])


def shift_channels(image: np.ndarray, shift: int, rng: random.Random,
                   angle: float | None = None) -> np.ndarray:
    """Décale les canaux rouge et bleu en sens opposés.

    La direction était tirée par le `random` global de l'original ; elle vient
    maintenant du tirage de la recette, pour qu'une séquence se rejoue. Un
    `angle` explicite la fixe : le décalage suit alors cette direction au
    lieu d'être tiré.
    """
    if shift <= 0:
        return image
    blue, green, red = cv2.split(image)

    if angle is not None:
        radians = np.deg2rad(angle)
        dx = int(round(shift * np.cos(radians)))
        dy = int(round(shift * np.sin(radians)))
        red = np.roll(np.roll(red, dx, axis=1), dy, axis=0)
        blue = np.roll(np.roll(blue, -dx, axis=1), -dy, axis=0)
        return cv2.merge([blue, green, red])

    if rng.random() < 0.7:  # horizontal sept fois sur dix, comme l'original
        red = np.roll(red, shift, axis=1)
        blue = np.roll(blue, -shift, axis=1)
    else:
        red = np.roll(red, shift // 2, axis=0)
        blue = np.roll(blue, -shift // 2, axis=0)
    return cv2.merge([blue, green, red])


def along_angle(image: np.ndarray, mask: np.ndarray, angle: float, work):
    """Applique une matière selon un axe incliné.

    Le tri et la trame travaillent par lignes ou par colonnes ; pour les
    incliner, on tourne l'image, on applique, et on remet d'aplomb. Le
    passage se fait dans un carré de la taille de la diagonale, sinon les
    coins sortiraient du cadre et reviendraient noirs. La double
    interpolation adoucit un peu le résultat : c'est le prix d'un angle
    libre, et il ne se paie qu'en dehors de l'horizontale.
    """
    if abs(angle) % 180.0 < 0.5:
        return work(image, mask)

    height, width = image.shape[:2]
    cote = int(np.ceil(np.hypot(height, width)))
    haut, gauche = (cote - height) // 2, (cote - width) // 2
    bas, droite = cote - height - haut, cote - width - gauche

    grand = cv2.copyMakeBorder(image, haut, bas, gauche, droite, cv2.BORDER_REFLECT)
    grand_masque = cv2.copyMakeBorder(
        mask.astype(np.uint8), haut, bas, gauche, droite, cv2.BORDER_CONSTANT, value=0
    )
    centre = (cote / 2.0, cote / 2.0)
    aller = cv2.getRotationMatrix2D(centre, angle, 1.0)
    retour = cv2.getRotationMatrix2D(centre, -angle, 1.0)

    tourne = cv2.warpAffine(grand, aller, (cote, cote),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    tourne_masque = cv2.warpAffine(grand_masque, aller, (cote, cote),
                                   flags=cv2.INTER_NEAREST) > 0

    fait = work(tourne, tourne_masque)
    remis = cv2.warpAffine(fait, retour, (cote, cote), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
    return remis[haut:haut + height, gauche:gauche + width]


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
    field, even = dissolve_field(saliency, even)
    steps = np.power(np.linspace(0.0, 1.0, max(frames, 2)), DWELL)
    thresholds = np.asarray(
        np.quantile(field, steps) if even else steps, dtype=np.float64
    )
    # Le dernier masque doit être vide, pour que la seconde image arrive
    # entièrement quelle que soit la distribution.
    thresholds[-1] = float(field.max()) + 1e-6
    return [(field >= t).astype(np.float32) for t in thresholds]


# Exposant appliqué à l'avancement avant d'en tirer un seuil : on s'attarde
# sur le milieu de la dissolution, là où se joue la décomposition.
DWELL = 0.8


def dissolve_field(saliency: np.ndarray, even: bool = True):
    """Prépare la carte sur laquelle la dissolution va mordre.

    Une carte sans relief — image unie, scan de papier blanc — rendrait tous
    les quantiles égaux, et rien ne se dissoudrait jamais. On se rabat alors
    sur un balayage spatial, qui donne au moins un mouvement. Le second
    renvoi dit si la répartition par quantiles reste applicable.
    """
    field = saliency.astype(np.float32)
    if field.max() > 1.0:
        field = field / 255.0
    field = cv2.GaussianBlur(field, (15, 15), 0)

    if float(field.max() - field.min()) < 1e-3:
        width = field.shape[1]
        field = np.tile(
            np.linspace(1.0, 0.0, width, dtype=np.float32), (field.shape[0], 1)
        )
        even = False
    return field, even


def retention_at(field: np.ndarray, progress: float, even: bool = True,
                 softness: float = 0.0) -> np.ndarray:
    """Ce qui reste d'une image à un avancement donné, entre 0 et 1.

    À 0 tout est gardé, à 1 il ne reste rien. Contrairement à
    `mask_sequence`, un seul instant est calculé : c'est ce qui permet de se
    déplacer dans le temps sans rejouer la séquence depuis le début.

    `field` est la carte préparée par `dissolve_field`, qui dit aussi si la
    répartition par quantiles reste applicable.

    Le masque est binaire par nature — un pixel est retiré ou il ne l'est
    pas — et découpe donc la matière au rasoir. `softness`, en pixels, fond
    cette découpe : la matière s'efface alors en s'amincissant, au lieu de
    s'arrêter net sur du noir.
    """
    step = float(np.clip(progress, 0.0, 1.0)) ** DWELL
    if step >= 1.0:
        return np.zeros(field.shape, dtype=np.float32)
    threshold = float(np.quantile(field, step)) if even else step
    garde = (field >= threshold).astype(np.float32)
    if softness > 0:
        garde = cv2.GaussianBlur(garde, (0, 0), float(softness))
    return garde


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
