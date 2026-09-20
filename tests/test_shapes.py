"""Les formes de trame, et le datamoshing posé sur une image arrêtée.

Une trame se ramène à un champ et à un seuil. Ces tests vérifient que chaque
forme est bien une forme — symétrique, centrée, croissante — et surtout que
toutes rendent la même densité au même réglage : sans quoi changer de forme
changerait aussi la clarté de l'image, et le curseur de grain ne voudrait
plus rien dire.

Le rond fait exception et garde le calcul de `blending-image2`, dont la
couverture croît comme le carré de la densité. C'est un écart à la théorie,
mais c'est lui qui fait les 230 œuvres ; `test_fidelity.py` le tient.
"""
import numpy as np
import pytest

from atelier.engine import (
    HALFTONE_SHAPES,
    Effect,
    Recipe,
    apply_halftone,
    rank_field,
    render,
    spot_field,
)
from atelier.metadata import read_recipe, save_with_recipe

AJOUTEES = tuple(s for s in HALFTONE_SHAPES if s != "round")


@pytest.fixture
def photo():
    rng = np.random.default_rng(4)
    return rng.integers(0, 255, (120, 96, 3), dtype=np.uint8)


@pytest.fixture
def degrade():
    """Un dégradé horizontal : toutes les densités dans une seule image."""
    bande = np.linspace(0, 255, 160, dtype=np.uint8)
    return np.dstack([np.tile(bande, (120, 1))] * 3)


@pytest.fixture
def sujet(degrade):
    """Un dégradé traversé de disques.

    Le dégradé seul ne suffit pas pour éprouver `render` : sa carte de
    saillance est presque nulle, les poids aussi, et la toile reste noire
    quelle que soit la trame. Il faut de la structure pour que la
    composition ait quelque chose à montrer.
    """
    import cv2
    image = degrade.copy()
    for x, rayon, couleur in ((40, 30, (240, 60, 60)), (110, 24, (40, 200, 90))):
        cv2.circle(image, (x, 60), rayon, couleur, -1)
    return image


class TestEachShapeIsAShape:
    @pytest.mark.parametrize("shape", HALFTONE_SHAPES)
    def test_the_cell_is_square(self, shape):
        """Une forme qui n'emploie qu'un axe doit tout de même remplir la cellule."""
        assert spot_field(16, shape).shape == (16, 16)

    @pytest.mark.parametrize("shape", HALFTONE_SHAPES)
    def test_the_centre_is_the_lowest_point(self, shape):
        field = spot_field(17, shape)
        assert field[8, 8] == pytest.approx(field.min(), abs=1e-6)

    @pytest.mark.parametrize("shape", AJOUTEES)
    def test_the_shape_grows_from_the_centre(self, shape):
        """Une densité plus forte ne doit jamais retirer de matière."""
        field = rank_field(16, shape)
        avant = field <= 0.3
        apres = field <= 0.6
        assert np.all(apres | ~avant), "la forme doit croître, pas se déplacer"

    @pytest.mark.parametrize("shape", AJOUTEES)
    def test_the_shape_is_symmetrical(self, shape):
        """Sans départage, les ex-aequo partiraient en escalier d'un côté."""
        plein = rank_field(17, shape) <= 0.5
        gauche_droite = np.mean(plein != plein[:, ::-1])
        haut_bas = np.mean(plein != plein[::-1, :])
        assert gauche_droite < 0.05, f"{shape} penche à droite ou à gauche"
        assert haut_bas < 0.05, f"{shape} penche en haut ou en bas"


class TestDensityIsHonoured:
    """Changer de forme ne doit pas changer la clarté de l'image."""

    @pytest.mark.parametrize("shape", AJOUTEES)
    @pytest.mark.parametrize("part", [0.25, 0.5, 0.75])
    def test_coverage_follows_the_level(self, shape, part):
        couvert = float(np.mean(rank_field(20, shape) <= part))
        assert couvert == pytest.approx(part, abs=0.03), \
            f"{shape} couvre {couvert:.0%} au lieu de {part:.0%}"

    @pytest.mark.parametrize("shape", AJOUTEES)
    def test_shapes_render_comparable_greys(self, degrade, shape):
        """Sur un même dégradé, deux formes doivent peser à peu près pareil."""
        plein = np.ones(degrade.shape[:2], bool)
        tramee = apply_halftone(degrade, plein, 8, shape)
        carre = apply_halftone(degrade, plein, 8, "square")
        ecart = abs(float(tramee.mean()) - float(carre.mean()))
        assert ecart < 18, f"{shape} s'écarte de {ecart:.0f} niveaux du carré"


class TestTheShapesDiffer:
    def test_every_shape_gives_a_different_image(self, degrade):
        """Sur un dégradé, donc à toutes les densités.

        Le test ne tient pas sur une image de bruit : ses cellules sont
        toutes à mi-densité, et à exactement 50 % le losange et
        l'euclidienne couvrent le même ensemble de pixels. C'est une
        coïncidence géométrique, pas une confusion des deux formes.
        """
        plein = np.ones(degrade.shape[:2], bool)
        rendus = {s: apply_halftone(degrade, plein, 10, s) for s in HALFTONE_SHAPES}
        for a in HALFTONE_SHAPES:
            for b in HALFTONE_SHAPES:
                if a < b:
                    assert not np.array_equal(rendus[a], rendus[b]), f"{a} == {b}"

    def test_an_unknown_shape_falls_back_to_round(self, photo):
        plein = np.ones(photo.shape[:2], bool)
        assert np.array_equal(
            apply_halftone(photo, plein, 8, "tarabiscotée"),
            apply_halftone(photo, plein, 8, "round"),
        )

    def test_line_is_horizontal(self):
        """Une trame ligne doit faire des bandes, pas des colonnes."""
        plein = rank_field(16, "line") <= 0.4
        assert plein.all(axis=1).sum() > 0, "des lignes entières doivent être pleines"
        assert plein.all(axis=0).sum() == 0, "aucune colonne ne doit l'être"

    def test_the_mask_is_still_honoured(self, photo):
        """Hors du masque, l'image ne bouge pas."""
        masque = np.zeros(photo.shape[:2], bool)
        masque[:, :40] = True
        out = apply_halftone(photo, masque, 6, "diamond")
        assert np.array_equal(out[:, 40:], photo[:, 40:])
        assert not np.array_equal(out[:, :40], photo[:, :40])


class TestTheShapeTravels:
    def test_the_shape_reaches_the_render(self, sujet):
        """Une cellule de 12 px : assez large pour porter une forme."""
        rendus = {}
        for shape in HALFTONE_SHAPES:
            recipe = Recipe(sources=("a",), seed=3,
                            effects=(Effect("halftone", 0.03, shape),))
            rendus[shape] = render(recipe, size=400, sources=[sujet])
        assert len({r.tobytes() for r in rendus.values()}) == len(HALFTONE_SHAPES)

    def test_a_tiny_cell_cannot_carry_a_shape(self, degrade):
        """Limite assumée : sous une poignée de pixels, les formes se valent.

        Neuf pixels ne suffisent pas à distinguer un carré d'un losange. Ce
        n'est pas un défaut à corriger — c'est ce que dit la géométrie — mais
        cela vaut d'être su : un grain très fin annule le choix de la forme.
        """
        plein = np.ones(degrade.shape[:2], bool)
        serres = {apply_halftone(degrade, plein, 3, s).tobytes()
                  for s in AJOUTEES}
        larges = {apply_halftone(degrade, plein, 14, s).tobytes()
                  for s in AJOUTEES}
        assert len(serres) < len(AJOUTEES)
        assert len(larges) == len(AJOUTEES)

    def test_the_shape_survives_the_metadata(self, tmp_path):
        recipe = Recipe(sources=("a.jpg",), seed=8,
                        effects=(Effect("halftone", 0.015, "diamond"),),
                        per_image=((Effect("halftone", 0.02, "cross"),),))
        image = np.zeros((32, 32, 3), np.uint8)
        back = read_recipe(save_with_recipe(image, tmp_path / "o.png", recipe))
        assert back.effects[0].shape == "diamond"
        assert back.per_image[0][0].shape == "cross"

    def test_older_images_keep_the_round_shape(self):
        """Une recette d'avant les formes doit rester une trame ronde."""
        from atelier.metadata import recipe_from_json
        back = recipe_from_json(
            '{"seed": 1, "effects": [{"name": "halftone", "strength": 0.01}]}')
        assert back.effects[0].shape == "round"

    def test_the_round_shape_is_not_written_out(self):
        """Le cas ordinaire n'alourdit pas les métadonnées."""
        from atelier.metadata import recipe_to_json
        recipe = Recipe(sources=("a",), seed=1, effects=(Effect("halftone", 0.01),))
        assert '"shape"' not in recipe_to_json(recipe)


class TestMoshingOnAStillImage:
    """Le tri canal par canal ne vivait que dans l'axe du temps."""

    @pytest.fixture
    def sources(self):
        rng = np.random.default_rng(7)
        return [rng.integers(0, 255, (150, 120, 3), dtype=np.uint8)]

    def test_it_changes_the_image(self, sources):
        nu = Recipe(sources=("a",), seed=5)
        moshe = Recipe(sources=("a",), seed=5, effects=(Effect("mosh", 0.03),))
        assert not np.array_equal(
            render(nu, size=160, sources=sources),
            render(moshe, size=160, sources=sources),
        )

    def test_it_separates_the_channels(self, sources):
        """C'est ce qui le distingue d'un tri ordinaire : les canaux divergent."""
        moshe = render(Recipe(sources=("a",), seed=5,
                              effects=(Effect("mosh", 0.04),)),
                       size=200, sources=sources)
        trie = render(Recipe(sources=("a",), seed=5,
                             effects=(Effect("pixelsort", 0.04),)),
                      size=200, sources=sources)
        assert not np.array_equal(moshe, trie)

    def test_it_is_reproducible(self, sources):
        recipe = Recipe(sources=("a",), seed=5, effects=(Effect("mosh", 0.03),))
        assert np.array_equal(
            render(recipe, size=160, sources=sources),
            render(recipe, size=160, sources=sources),
        )

    def test_two_seeds_diverge(self, sources):
        a = render(Recipe(sources=("a",), seed=1, effects=(Effect("mosh", 0.03),)),
                   size=160, sources=sources)
        b = render(Recipe(sources=("a",), seed=2, effects=(Effect("mosh", 0.03),)),
                   size=160, sources=sources)
        assert not np.array_equal(a, b)

    def test_it_scales_with_the_canvas(self, sources):
        """Comme toutes les matières, sa force est une fraction de la toile."""
        recipe = Recipe(sources=("a",), seed=5, effects=(Effect("mosh", 0.03),))
        assert render(recipe, size=300, sources=sources).shape == (300, 300, 3)
