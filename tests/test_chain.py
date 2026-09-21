"""L'enchaînement, et ce qui le distingue de la pile.

Les vidéos de février 2025 ne composent rien : à chaque instant il n'y a
qu'une image, ou deux en train de se relayer, et ce que la première
abandonne, la seconde l'occupe entièrement. C'est ce qui explique qu'on n'y
voie jamais de fond — la couverture vaut exactement 1 partout, à tout
instant. La pile, elle, finit toujours par déboucher sur du noir quand
l'évidement a mangé la matière.

Ces tests tiennent les deux propriétés qui font le mode : la toile reste
pleine du début à la fin, et chaque transition arrive vraiment au bout,
sinon l'image suivante n'apparaîtrait jamais entière.
"""
import glob

import cv2
import numpy as np
import pytest

from atelier.chain import compte, par_paire, prepare
from atelier.decay import Decay, compose_at, count_frames
from atelier.engine import Effect, Recipe
from atelier.metadata import read_recipe, save_with_recipe
from atelier.video import write_video

TAILLE = 288


@pytest.fixture(scope="module")
def photos():
    chemins = sorted(glob.glob("compost-test/*.jpg"))[:4]
    if len(chemins) < 3:
        pytest.skip("il faut au moins trois photographies pour enchaîner")
    return tuple(chemins), [cv2.imread(p) for p in chemins]


def recette(photos, **extra) -> Recipe:
    return Recipe(sources=photos[0], seed=808, full_frame=True, **extra)


def axe(frames: int = 36, **extra) -> Decay:
    return Decay(frames=frames, chained=True, **extra)


def part_noire(image: np.ndarray) -> float:
    return float((image.max(axis=2) < 10).mean())


def toutes(recipe, decay, photos):
    total = count_frames(decay, len(photos[1]))
    return [compose_at(recipe, decay, TAILLE, i, sources=photos[1])
            for i in range(total)]


def test_la_toile_reste_pleine_tout_du_long(photos):
    """La propriété qui fait le mode : jamais de fond, à aucun instant."""
    noirs = [part_noire(f) for f in toutes(recette(photos), axe(), photos)]
    assert max(noirs) < 0.005, max(noirs)


def test_la_pile_au_contraire_finit_par_creuser(photos):
    """Le contraste qui justifie d'avoir deux modes et non un réglage."""
    recipe, pile = recette(photos), Decay(frames=36)
    total = count_frames(pile, len(photos[1]))
    fin = compose_at(recipe, pile, TAILLE, total - 1, sources=photos[1])
    assert part_noire(fin) > 0.4


def test_chaque_transition_arrive_au_bout(photos):
    """La dernière image d'un enchaînement est la suivante, intacte.

    Sans cela, l'image d'après n'apparaîtrait jamais entière et le montage
    ne monterait rien. C'est aussi ce qui raccorde les transitions : la
    suivante repart de cette même image.
    """
    recipe, decay = recette(photos), axe()
    toiles, _, _ = prepare(recipe, TAILLE, photos[1])
    longueur = par_paire(decay, len(toiles))

    for paire in range(len(toiles) - 1):
        derniere = compose_at(recipe, decay, TAILLE,
                              (paire + 1) * longueur - 1, sources=photos[1])
        assert np.array_equal(derniere, toiles[paire + 1]), paire


def test_le_premier_instant_montre_l_image_telle_quelle(photos):
    """Comme pour la pile : on voit ce qu'on a réglé avant de le voir se défaire."""
    recipe = recette(photos)
    toiles, _, _ = prepare(recipe, TAILLE, photos[1])
    premiere = compose_at(recipe, axe(), TAILLE, 0, sources=photos[1])
    assert np.array_equal(premiere, toiles[0])


def test_une_image_de_plus_n_allonge_pas_la_video(photos):
    """L'axe garde sa longueur et se partage : ajouter une image resserre.

    Une vidéo qui doublerait de durée parce qu'on a posé une image de plus
    est une surprise qu'on ne veut pas au moment du tirage.
    """
    decay = axe()
    assert compte(decay, 2) == 36
    assert compte(decay, 3) == 36
    assert compte(decay, 4) == 36
    # Au nombre entier près : 36 ne se divise pas en cinq.
    assert abs(compte(decay, 6) - 36) <= 5


def test_l_axe_reste_parcourable_dans_le_desordre(photos):
    """Une frame ne dépend pas des précédentes : on se déplace librement."""
    recipe, decay = recette(photos), axe()
    seule = compose_at(recipe, decay, TAILLE, 14, sources=photos[1])
    apres = toutes(recipe, decay, photos)[14]
    assert np.array_equal(seule, apres)


def test_les_matieres_posees_s_appliquent(photos):
    """Une trame choisie doit se voir dans l'enchaînement comme sur la pile."""
    nue = compose_at(recette(photos), axe(), TAILLE, 8, sources=photos[1])
    tramee = compose_at(
        recette(photos, effects=(Effect("halftone", 0.012),)),
        axe(), TAILLE, 8, sources=photos[1],
    )
    ecart = float(np.abs(nue.astype(np.int16) - tramee.astype(np.int16)).mean())
    assert ecart > 8, ecart


def test_le_mode_voyage_avec_l_image(photos, tmp_path):
    """Un instant prélevé doit se rouvrir dans le mode qui l'a produit."""
    from atelier.metadata import read_payload
    from atelier.server import decay_payload

    recipe, decay = recette(photos), axe()
    chemin = save_with_recipe(
        compose_at(recipe, decay, 120, 5, sources=photos[1]),
        tmp_path / "essai.jpg", recipe, moment=5, **decay_payload(decay),
    )
    assert read_payload(chemin)["chained"] is True
    assert read_recipe(chemin).seed == recipe.seed


def test_une_seule_image_ne_s_enchaine_pas(photos):
    """Il faut deux images pour une transition : le dire plutôt que planter."""
    seule = Recipe(sources=(photos[0][0],), seed=1, full_frame=True)
    with pytest.raises(ValueError):
        compose_at(seule, axe(), TAILLE, 0, sources=photos[1][:1])


def test_la_video_s_ecrit(photos, tmp_path):
    """Le tirage vidéo doit passer par le mode sans traitement particulier."""
    from atelier.decay import sequence

    cible = tmp_path / "essai.mp4"
    ecrites = write_video(sequence(recette(photos), axe(frames=8), 160,
                                   photos[1]), cible, 12)
    assert ecrites == count_frames(axe(frames=8), len(photos[1]))
    assert cible.exists() and cible.stat().st_size > 0
