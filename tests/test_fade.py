"""La saillance comme présence, et non seulement comme arbitre.

La composition divise la toile par la somme des poids, ce qui lui rend sa
pleine intensité partout où une couche est posée. Sans cette division il ne
resterait que quelques points allumés : la carte de saillance est si
concentrée que le poids total reste sous 1 sur plus de quatre-vingt-dix-neuf
pour cent de la toile. Mais elle a une conséquence qu'on ne voit qu'à
l'usage — la saillance ne décide plus que de qui l'emporte entre deux
couches, et n'efface plus rien. En plein cadre, où tout est couvert, une
séquence entière peut se dérouler sans un seul noir, et l'évidement du temps
ne fait que découvrir la couche du dessous.

`fade` rouvre cette possibilité. Ces tests vérifient qu'elle est exacte —
la part demandée est bien celle qui s'éteint —, qu'elle laisse la
composition intacte à zéro, et qu'elle donne au temps de quoi creuser.
"""
import glob

import cv2
import numpy as np
import pytest

from atelier.engine import Effect, Recipe, normaliser, render
from atelier.decay import Decay, compose_at
from atelier.metadata import read_recipe, save_with_recipe

TAILLE = 320


@pytest.fixture(scope="module")
def photos():
    chemins = sorted(glob.glob("compost-test/*.jpg"))[:3]
    if len(chemins) < 2:
        pytest.skip("il faut des photographies pour mesurer une présence")
    return tuple(chemins), [cv2.imread(p) for p in chemins]


def recette(photos, **extra) -> Recipe:
    return Recipe(sources=photos[0], seed=404, full_frame=True, **extra)


def part_noire(image: np.ndarray) -> float:
    """La part de la toile qui ne porte plus rien de visible."""
    return float((image.max(axis=2) < 10).mean())


def test_le_plein_cadre_ne_laisse_aucun_noir_sans_disparition(photos):
    """Le constat d'où vient le réglage : une toile couverte n'a plus de fond."""
    image = render(recette(photos), size=TAILLE, sources=photos[1])
    assert part_noire(image) < 0.01


def test_la_part_qui_s_efface_est_celle_qu_on_demande(photos):
    """Le réglage est un centile, comme le seuil des matières.

    Une valeur brute ne dirait rien : la carte de saillance est trop
    concentrée pour qu'un seuil en absolu se comporte pareil d'une image à
    l'autre. À 40 %, ce sont les quarante pour cent les moins présents de la
    toile qui s'éteignent, quelles que soient les photographies.
    """
    for demande in (0.25, 0.5, 0.75):
        image = render(recette(photos, fade=demande), size=TAILLE,
                       sources=photos[1])
        assert abs(part_noire(image) - demande) < 0.06, demande


def test_a_zero_la_composition_est_inchangee(photos):
    """Le réglage neutre doit rendre exactement l'image d'avant lui."""
    sans = render(recette(photos), size=TAILLE, sources=photos[1])
    zero = render(recette(photos, fade=0.0), size=TAILLE, sources=photos[1])
    assert np.array_equal(sans, zero)


def test_le_bord_du_noir_est_un_degrade_et_non_une_coupure(photos):
    """La disparition ne doit pas découper la toile comme un masque binaire."""
    image = render(recette(photos, fade=0.5), size=TAILLE, sources=photos[1])
    clair = image.max(axis=2)
    entre_deux = float(((clair > 10) & (clair < 120)).mean())
    assert entre_deux > 0.05


def test_l_evidement_ne_creuse_qu_a_la_toute_fin_sans_disparition(photos):
    """Le constat qui a fait ce réglage, mesuré sur l'axe entier.

    L'évidement finit bien par ouvrir du noir, mais seulement quand il a
    mangé presque toute la matière : sur douze images en plein cadre, la
    moitié du parcours se passe sans que la saillance efface quoi que ce
    soit — ce qui la découvre est remplacé par la couche du dessous, à
    laquelle la normalisation rend sa pleine intensité. La disparition
    l'ouvre dès le premier instant, et le creusement se voit sur toute la
    durée au lieu de tomber dans les dernières secondes.
    """
    axe = Decay(frames=12)
    milieu = 5
    pleine = compose_at(recette(photos), axe, TAILLE, milieu, sources=photos[1])
    creuse = compose_at(recette(photos, fade=0.4), axe, TAILLE, milieu,
                        sources=photos[1])

    assert part_noire(pleine) < 0.15
    assert part_noire(creuse) > part_noire(pleine) + 0.25


def test_le_noir_gagne_le_long_de_l_axe(photos):
    """La toile doit se vider en avançant, pas s'éteindre d'un coup."""
    axe = Decay(frames=12)
    recipe = recette(photos, fade=0.35)
    parts = [part_noire(compose_at(recipe, axe, TAILLE, i, sources=photos[1]))
             for i in (0, 6, 11)]
    assert parts[0] < parts[1] < parts[2]


def test_la_disparition_voyage_avec_la_recette(photos, tmp_path):
    """Un tirage retenu doit se rouvrir tel qu'il est sorti."""
    recipe = recette(photos, fade=0.42, effects=(Effect("halftone", 0.01),))
    chemin = save_with_recipe(
        render(recipe, size=120, sources=photos[1]),
        tmp_path / "essai.jpg", recipe,
    )
    assert read_recipe(chemin).fade == pytest.approx(0.42)


def test_la_normalisation_rend_son_intensite_a_ce_qui_reste():
    """Le fondu s'applique après la division, pas à sa place.

    Une couche posée faiblement doit sortir à sa vraie couleur là où elle
    reste, sinon la disparition assombrirait toute l'image au lieu d'en
    creuser une part.
    """
    canvas = np.zeros((4, 4, 3), np.float32)
    weights = np.full((4, 4), 0.25, np.float32)
    canvas[:] = 200.0 * 0.25
    normaliser(canvas, weights, 0.0)
    assert np.allclose(canvas, 200.0)
