"""Un tirage au hasard, réglé sur les œuvres conservées.

Le bouton « Au hasard » ne tire pas uniformément : il tire dans ce que le
corpus de 2024 laisse voir des habitudes de travail. Ces chiffres ne sont pas
des préférences supposées, ils viennent de mesures.

## Ce que dit le corpus

Les 230 œuvres de `le corpus conservé de 2024/` ont été mesurées sur quatre
grandeurs : contraste (écart-type des gris), saturation moyenne, part de
pixels en noir ou blanc purs, et énergie haute fréquence (laplacien).

Deux séries étaient déjà rangées par technique — c'est l'étiquetage de Tom
lui-même, et il sert de vérité terrain :

| famille  | contraste | saturation | extrêmes | détail |
|----------|-----------|------------|----------|--------|
| bitmap   | 82.6      | 39.1       | 62.5 %   | 19.3   |
| halftone | 75.2      | 47.5       | 65.7 %   | 114.6  |
| pixelate | 67.9      | 78.2       | 36.3 %   | 7.6    |
| saturate | 55.8      | 95.5       | 38.5 %   | 20.1   |

Les quatre séries non rangées ont été classées sur ces centres. Résultat sur
l'ensemble : **trame 53 %, bitmap 23 %, pixellisation 17 %, saturation 7 %**.
La trame domine largement ce qui a été gardé.

Deux traits reviennent partout : un contraste élevé, autour de 75, et plus
de la moitié des pixels en noir ou blanc purs. Ce n'est pas un travail en
demi-teintes.

## Comment les plages ont été trouvées

Les plages ci-dessous ont été réglées en tirant des rendus en 1024 px — la
taille du corpus, car le laplacien n'est pas invariant à l'échelle — puis en
les mesurant contre ces centres. Deux réglages viennent directement des
chiffres : le bitmap ne descend pas sous 0.85, sinon le seuillage reste mêlé
à l'image et le noir et blanc francs du corpus n'apparaît pas ; et le
débordement de saturation, qui ne concerne que 16 œuvres sur 230, ne se tire
qu'une fois sur sept.

## Ce qui reste en écart, et pourquoi on s'y arrête

Sur un tirage pondéré mesuré à 1024 px, contre le corpus :

| grandeur   | tiré | corpus |
|------------|------|--------|
| contraste  | 81.4 | 74.8   |
| saturation | 53.0 | 53.9   |
| extrêmes   | 47.0 | 53.0   |
| détail     | 74.4 | 53.0   |

Deux remarques sur le détail. D'abord la cible globale de 53 est en tension
avec les centres par famille, qui pondérés en donneraient 68 : les centres
viennent des deux séries étiquetées, la moyenne des 230. Ensuite la famille
bitmap mesure 55 contre 19.3 attendu, et cet écart tient surtout aux images
de départ : les œuvres bitmap conservées partent d'images bien plus douces
que les photos d'essai. Serrer davantage reviendrait à régler le tirage sur
un corpus fait d'autres sources — donc sur du bruit.
"""

from __future__ import annotations

import random

from atelier.engine import Effect, Recipe
from atelier.video import Motion

# Part de chaque famille dans les 230 œuvres conservées
FAMILIES = ("halftone", "bitmap", "pixelate", "saturate")
WEIGHTS = (53.5, 22.6, 17.0, 7.0)

# Par famille : la matière dominante, toujours présente, puis une compagne
# occasionnelle, et la part de l'image livrée aux matières.
PROFILES = {
    "halftone": {
        "always": ("halftone", 0.010, 0.018),
        "sometimes": (0.55, "bitmap", 0.55, 0.95),
        "coverage": (80, 96),
    },
    "bitmap": {
        # Le halftone reste rare ici : il porte à lui seul l'essentiel du
        # détail, et il noyait la famille bitmap, qui est la plus sobre.
        # En dessous de 0.85 le seuillage reste mélangé à l'image d'origine et
        # ne donne pas le noir et blanc francs que montre le corpus.
        "always": ("bitmap", 0.85, 1.00),
        "sometimes": (0.12, "halftone", 0.020, 0.035),
        "coverage": (75, 95),
    },
    "pixelate": {
        "always": ("pixelate", 0.010, 0.032),
        "sometimes": (0.65, "saturate", 1.40, 2.60),
        "coverage": (45, 72),
    },
    "saturate": {
        "always": ("saturate", 2.20, 4.00),
        "sometimes": (0.55, "pixelate", 0.010, 0.020),
        "coverage": (40, 68),
    },
}

# Les deux matières venues de la voie vidéo n'existaient pas en 2024 : elles
# apparaissent rarement, pour ouvrir sans dénaturer.
NEWCOMERS = ((0.12, "pixelsort", 0.02, 0.08), (0.10, "shift", 0.004, 0.015))

SOURCES_RANGE = (2, 4)


def draw_effects(family: str, rng: random.Random) -> tuple[Effect, ...]:
    """Les matières d'une famille : la dominante, et parfois sa compagne."""
    profile = PROFILES[family]
    name, low, high = profile["always"]
    effects = [Effect(name, rng.uniform(low, high))]

    chance, companion, c_low, c_high = profile["sometimes"]
    if rng.random() < chance:
        effects.append(Effect(companion, rng.uniform(c_low, c_high)))

    for chance, name, low, high in NEWCOMERS:
        if rng.random() < chance:
            effects.append(Effect(name, rng.uniform(low, high)))

    # L'ordre compte : la trame écrase ce qui la précède, on la garde en
    # dernier pour que la pixellisation reste lisible dessous.
    order = {n: i for i, n in enumerate(
        ("saturate", "pixelate", "pixelsort", "shift", "bitmap", "halftone"))}
    effects.sort(key=lambda e: order.get(e.name, 99))
    return tuple(effects)


def random_recipe(available: list[str], rng: random.Random | None = None,
                  family: str | None = None) -> Recipe:
    """Tire une recette dans les habitudes que révèle le corpus."""
    rng = rng or random.Random()
    if not available:
        raise ValueError("aucune image disponible")

    family = family or rng.choices(FAMILIES, weights=WEIGHTS, k=1)[0]
    count = min(len(available), rng.randint(*SOURCES_RANGE))
    chosen = rng.sample(available, count)

    low, high = PROFILES[family]["coverage"]
    effects = draw_effects(family, rng)

    # Une image sur quatre environ reçoit un réglage à elle : c'est ce qui
    # évite que la composition paraisse traitée d'un seul bloc. La famille du
    # tirage reste largement favorisée, sinon chaque image tirerait de son
    # côté et la composition perdrait son unité.
    per_image = tuple(
        draw_effects(family if rng.random() < 0.6
                     else rng.choices(FAMILIES, weights=WEIGHTS, k=1)[0], rng)
        if rng.random() < 0.25 else None
        for _ in chosen
    )

    return Recipe(
        sources=tuple(chosen),
        # La graine vient du tirage : un rng donné rejoue le même tirage entier.
        seed=rng.randrange(2 ** 31),
        effects=effects,
        per_image=per_image,
        saliency_threshold=rng.randint(low, high),
        smoothness=rng.uniform(1.5, 6.0),
        edge_blur=0 if rng.random() < 0.8 else rng.randint(2, 8),
        # Le débordement de saturation est un accident devenu matière, mais il
        # ne touche que 16 œuvres sur 230 : il reste l'exception.
        saturation_overflow=rng.random() < 0.15,
    )


def random_motion(rng: random.Random | None = None) -> Motion:
    """Un mouvement plausible pour la voie vidéo.

    Les vidéos conservées durent de une à douze secondes, à 20 ou 24 images
    par seconde, ce qui place une transition entre 16 et 40 images.
    """
    rng = rng or random.Random()
    return Motion(
        frames_per_transition=rng.randint(16, 40),
        fps=rng.choice((20, 24)),
        min_segment=rng.uniform(0.02, 0.05),
        max_segment=rng.uniform(0.08, 0.16),
        channel_shift=rng.uniform(0.008, 0.025),
        enhanced=rng.random() < 0.8,
    )
