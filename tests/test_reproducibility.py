"""La graine sert à une seule chose : reprendre un tirage pour l'agrandir.

Ces tests vérifient les deux moitiés de cette phrase. D'une part le hasard
reste entier — sans graine imposée, deux rendus diffèrent. D'autre part une
graine donnée produit la même composition à n'importe quelle taille, sinon
elle ne servirait à rien.
"""
import cv2
import numpy as np
import pytest

from atelier.engine import (
    Effect,
    Recipe,
    new_seed,
    placements_for,
    render,
)


@pytest.fixture
def sources():
    """Trois images nettement distinctes, pour voir la composition bouger."""
    def disc(colour, radius):
        image = np.zeros((400, 300, 3), dtype=np.uint8)
        cv2.circle(image, (150, 200), radius, colour, -1)
        return image

    return [disc((40, 60, 220), 120), disc((220, 80, 40), 90), disc((60, 200, 90), 60)]


@pytest.fixture
def recipe():
    return Recipe(
        sources=("a.jpg", "b.jpg", "c.jpg"),
        seed=1234,
        effects=(Effect("pixelate", 0.008), Effect("bitmap", 0.5)),
    )


def similarity(a, b):
    """Ressemblance de deux rendus, ramenés à une même petite taille."""
    small = (128, 128)
    ga = cv2.cvtColor(cv2.resize(a, small, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, small, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return 1.0 - np.mean(np.abs(ga.astype(float) - gb.astype(float))) / 255.0


class TestTheSeedSurvivesResolution:
    """Le point de tout l'exercice."""

    def test_same_recipe_two_sizes_same_composition(self, recipe, sources):
        small = render(recipe, size=256, sources=sources)
        large = render(recipe, size=1024, sources=sources)
        assert similarity(small, large) > 0.95, "la composition doit se retrouver"

    def test_placements_do_not_depend_on_size(self, recipe):
        """Les positions sont tirées en fractions, pas en pixels."""
        first = placements_for(recipe)
        second = placements_for(recipe)
        assert [(p.fx, p.fy) for p in first] == [(p.fx, p.fy) for p in second]
        assert all(0.0 <= p.fx <= 1.0 and 0.0 <= p.fy <= 1.0 for p in first)

    def test_rendering_twice_is_identical(self, recipe, sources):
        a = render(recipe, size=256, sources=sources)
        b = render(recipe, size=256, sources=sources)
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("size", [256, 512, 1024])
    def test_output_is_square_and_sized(self, recipe, sources, size):
        assert render(recipe, size=size, sources=sources).shape == (size, size, 3)


class TestTheChanceRemains:
    """Une graine ne doit rien figer d'autre que ce qu'on lui demande."""

    def test_two_seeds_give_two_images(self, recipe, sources):
        a = render(recipe.with_seed(1), size=256, sources=sources)
        b = render(recipe.with_seed(2), size=256, sources=sources)
        assert similarity(a, b) < 0.98, "deux graines doivent composer autrement"

    def test_new_seed_varies(self):
        seeds = {new_seed() for _ in range(50)}
        assert len(seeds) > 45, "le tirage doit rester du hasard"


class TestEffectsScaleWithTheCanvas:
    """Une force en pixels donnerait un grain quatre fois plus fin en grand."""

    def test_strength_grows_with_size(self):
        effect = Effect("halftone", 0.01)
        assert effect.pixels(1024) == 10
        assert effect.pixels(6000) == 60

    def test_never_degenerates_to_zero(self):
        assert Effect("pixelate", 0.0001).pixels(256) >= 2

    def test_grain_is_visually_comparable_across_sizes(self, sources):
        """Le grain de la trame doit occuper la même part de l'image."""
        recipe = Recipe(sources=("a.jpg",), seed=7,
                        effects=(Effect("halftone", 0.02),))
        small = render(recipe, size=256, sources=sources[:1])
        large = render(recipe, size=1024, sources=sources[:1])
        assert similarity(small, large) > 0.90


class TestRobustness:
    def test_no_sources_is_refused(self):
        with pytest.raises(ValueError):
            render(Recipe(sources=(), seed=1), size=128, sources=[])

    def test_unreadable_source_is_refused(self):
        with pytest.raises(ValueError):
            render(Recipe(sources=("/introuvable.jpg",), seed=1), size=128)

    def test_recipe_without_effects_still_composes(self, sources):
        out = render(Recipe(sources=("a.jpg",), seed=3), size=128, sources=sources[:1])
        assert out.shape == (128, 128, 3)
        assert out.any(), "la toile ne doit pas rester vide"

    def test_source_larger_than_canvas(self, sources):
        """Une photo de 6000 px sur une toile de 256 doit tenir."""
        big = cv2.resize(sources[0], (3000, 2400))
        out = render(Recipe(sources=("a.jpg",), seed=3), size=256, sources=[big])
        assert out.shape == (256, 256, 3)


class TestPerImageEffects:
    """Chaque image peut recevoir sa propre matière."""

    def test_an_override_replaces_the_common_setting(self, sources):
        from atelier.engine import effects_for
        recipe = Recipe(
            sources=("a", "b", "c"), seed=5,
            effects=(Effect("bitmap", 0.7),),
            per_image=(None, (Effect("halftone", 0.01),), None),
        )
        assert [e.name for e in effects_for(recipe, 0)] == ["bitmap"]
        assert [e.name for e in effects_for(recipe, 1)] == ["halftone"]
        assert [e.name for e in effects_for(recipe, 2)] == ["bitmap"]

    def test_an_empty_override_means_no_effect(self):
        """Laisser une image intacte pendant que les autres se dégradent."""
        from atelier.engine import effects_for
        recipe = Recipe(sources=("a", "b"), seed=5,
                        effects=(Effect("bitmap", 0.7),), per_image=(None, ()))
        assert effects_for(recipe, 1) == ()

    def test_missing_entries_fall_back(self):
        from atelier.engine import effects_for
        recipe = Recipe(sources=("a", "b", "c"), seed=5,
                        effects=(Effect("bitmap", 0.7),), per_image=(None,))
        assert [e.name for e in effects_for(recipe, 2)] == ["bitmap"]

    def test_the_render_actually_differs(self, sources):
        commun = Recipe(sources=("a", "b", "c"), seed=5,
                        effects=(Effect("bitmap", 0.9),))
        mixte = Recipe(sources=("a", "b", "c"), seed=5,
                       effects=(Effect("bitmap", 0.9),),
                       per_image=(None, (Effect("halftone", 0.02),), None))
        assert not np.array_equal(
            render(commun, size=256, sources=sources),
            render(mixte, size=256, sources=sources),
        )
