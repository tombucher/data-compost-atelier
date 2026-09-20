"""L'axe du temps : une composition unique, qui se décompose.

Le test qui compte est le premier : l'instant 0 doit rendre exactement ce que
rendait la voie image. Sans lui, fusionner les deux voies reviendrait à perdre
les œuvres de 2024 en route.

Les autres vérifient que l'axe est bien un axe — qu'il avance, que chaque
instant se calcule seul, qu'il ne dépend pas de la taille du rendu — et que
les réglages exposés dans l'interface font ce qu'ils annoncent.
"""
import random

import cv2
import numpy as np
import pytest

from atelier.chance import random_decay
from atelier.decay import (
    Decay,
    compose_at,
    count_frames,
    layer_progress,
    sequence,
)
from atelier.engine import Effect, Recipe, render


@pytest.fixture
def sources():
    """Trois images nettement distinctes, pour voir la pile bouger."""
    def disc(colour, radius):
        image = np.zeros((300, 240, 3), np.uint8)
        cv2.circle(image, (120, 150), radius, colour, -1)
        return image
    return [disc((30, 60, 220), 100), disc((220, 90, 40), 75), disc((60, 210, 90), 50)]


@pytest.fixture
def recipe():
    return Recipe(
        sources=("a.jpg", "b.jpg", "c.jpg"), seed=1234,
        effects=(Effect("halftone", 0.012),),
    )


def ressemblance(a, b):
    """Ressemblance de deux rendus, ramenés à une même petite taille."""
    petite = (96, 96)
    ga = cv2.cvtColor(cv2.resize(a, petite, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, petite, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return 1.0 - np.mean(np.abs(ga.astype(float) - gb.astype(float))) / 255.0


class TestTheStartLosesNothing:
    """Le point de toute la fusion : l'instant 0 est la composition d'avant."""

    def test_instant_zero_is_the_still_render(self, recipe, sources):
        fixe = render(recipe, size=256, sources=sources)
        zero = compose_at(recipe, Decay(), 256, 0, sources)
        assert np.array_equal(fixe, zero), "l'axe doit partir de l'image intacte"

    @pytest.mark.parametrize("reglage", [
        {"stagger": 0.0}, {"stagger": 1.0}, {"residue": 0.0}, {"residue": 0.5},
        {"channel_shift": 0.1}, {"enhanced": False}, {"sort_visible": False},
        {"frames": 8}, {"frames": 200},
    ])
    def test_no_setting_disturbs_the_start(self, recipe, sources, reglage):
        """Aucun réglage de décomposition ne doit mordre à l'instant 0."""
        fixe = render(recipe, size=192, sources=sources)
        zero = compose_at(recipe, Decay(**reglage), 192, 0, sources)
        assert np.array_equal(fixe, zero), f"{reglage} entame déjà l'image"

    def test_a_single_image_still_composes(self, recipe, sources):
        """Une pile d'une seule couche reste une composition valable."""
        seule = Recipe(sources=("a.jpg",), seed=3)
        out = compose_at(seule, Decay(), 128, 0, sources[:1])
        assert out.shape == (128, 128, 3)
        assert out.any()


class TestTheAxisAdvances:
    def test_the_composition_comes_undone(self, recipe, sources):
        decay = Decay(frames=24)
        debut = compose_at(recipe, decay, 200, 0, sources)
        fin = compose_at(recipe, decay, 200, 23, sources)
        assert not np.array_equal(debut, fin)
        assert ressemblance(debut, fin) < 0.9, "la fin doit être visiblement défaite"

    def test_matter_disappears_rather_than_appears(self, recipe, sources):
        """Composter, c'est perdre de la matière, pas en gagner."""
        decay = Decay(frames=24, residue=0.05)
        vides = [
            float(np.mean(cv2.cvtColor(
                compose_at(recipe, decay, 160, i, sources), cv2.COLOR_BGR2GRAY) < 8))
            for i in (0, 12, 23)
        ]
        assert vides[0] < vides[1] < vides[2], f"le vide doit gagner : {vides}"

    def test_nothing_is_left_when_residue_is_zero(self, recipe, sources):
        decay = Decay(frames=12, residue=0.0, stagger=0.0)
        fin = compose_at(recipe, decay, 128, 11, sources)
        assert fin.max() == 0, "sans résidu, l'axe doit s'éteindre au noir"

    def test_residue_keeps_something(self, recipe, sources):
        decay = Decay(frames=12, residue=0.3, stagger=0.0)
        fin = compose_at(recipe, decay, 128, 11, sources)
        assert fin.any(), "avec un résidu, il doit rester de la matière"


class TestEveryMomentStandsAlone:
    """Se déplacer dans le temps sans rejouer la séquence."""

    def test_a_moment_matches_the_sequence(self, recipe, sources):
        decay = Decay(frames=6)
        entier = list(sequence(recipe, decay, 128, sources))
        for index in (0, 2, 5):
            seul = compose_at(recipe, decay, 128, index, sources)
            assert np.array_equal(seul, entier[index]), f"instant {index}"

    def test_the_sequence_has_the_announced_length(self, recipe, sources):
        decay = Decay(frames=7)
        assert len(list(sequence(recipe, decay, 96, sources))) == count_frames(decay) == 7

    def test_the_length_ignores_the_number_of_images(self, sources):
        """Une image de plus est une couche, pas une diapositive."""
        decay = Decay(frames=10)
        deux = Recipe(sources=("a", "b"), seed=1)
        trois = Recipe(sources=("a", "b", "c"), seed=1)
        assert len(list(sequence(deux, decay, 96, sources[:2]))) == \
               len(list(sequence(trois, decay, 96, sources))) == 10

    def test_the_index_is_clamped(self, recipe, sources):
        decay = Decay(frames=8)
        assert compose_at(recipe, decay, 96, 999, sources).shape == (96, 96, 3)
        assert np.array_equal(
            compose_at(recipe, decay, 96, -5, sources),
            compose_at(recipe, decay, 96, 0, sources),
        )

    def test_replaying_gives_the_same_axis(self, recipe, sources):
        decay = Decay(frames=5)
        a = list(sequence(recipe, decay, 96, sources))
        b = list(sequence(recipe, decay, 96, sources))
        assert all(np.array_equal(x, y) for x, y in zip(a, b))

    def test_two_seeds_diverge(self, sources):
        decay = Decay(frames=5)
        a = compose_at(Recipe(sources=("a", "b", "c"), seed=1), decay, 128, 3, sources)
        b = compose_at(Recipe(sources=("a", "b", "c"), seed=2), decay, 128, 3, sources)
        assert not np.array_equal(a, b)


class TestTheAxisSurvivesResolution:
    """Prélever un instant en grand, c'est tout l'intérêt de l'axe."""

    @pytest.mark.parametrize("size", [128, 256, 512])
    def test_output_is_square_and_sized(self, recipe, sources, size):
        assert compose_at(recipe, Decay(), size, 4, sources).shape == (size, size, 3)

    def test_the_same_moment_is_recognisable_at_two_sizes(self, recipe, sources):
        decay = Decay(frames=16)
        petite = compose_at(recipe, decay, 200, 5, sources)
        grande = compose_at(recipe, decay, 600, 5, sources)
        assert ressemblance(petite, grande) > 0.75, "l'instant doit se reconnaître"


class TestTheLayersComeUndoneInTurn:
    def test_the_last_chosen_goes_first(self):
        """Ce qui est posé en dernier est le plus exposé."""
        decay = Decay(stagger=0.6)
        dessus = layer_progress(decay, 0.2, 2, 3)
        dessous = layer_progress(decay, 0.2, 0, 3)
        assert dessus > dessous

    def test_without_stagger_everything_rots_together(self):
        decay = Decay(stagger=0.0)
        parts = {layer_progress(decay, 0.4, k, 4) for k in range(4)}
        assert parts == {0.4}

    def test_every_layer_is_done_at_the_end(self):
        for stagger in (0.0, 0.3, 0.7, 1.0):
            decay = Decay(stagger=stagger)
            assert all(layer_progress(decay, 1.0, k, 4) == 1.0 for k in range(4))

    def test_progress_never_leaves_its_bounds(self):
        decay = Decay(stagger=0.8)
        for moment in np.linspace(0, 1, 21):
            for k in range(5):
                assert 0.0 <= layer_progress(decay, float(moment), k, 5) <= 1.0

    def test_staggering_changes_the_middle_of_the_axis(self, recipe, sources):
        ensemble = compose_at(recipe, Decay(frames=20, stagger=0.0), 160, 10, sources)
        tour_a_tour = compose_at(recipe, Decay(frames=20, stagger=0.9), 160, 10, sources)
        assert not np.array_equal(ensemble, tour_a_tour)


class TestTheGlitchSettingsBite:
    """Chaque réglage exposé dans l'interface doit changer quelque chose."""

    @pytest.mark.parametrize("reglage", [
        {"min_segment": 0.12}, {"max_segment": 0.35}, {"segment_growth": 4.0},
        {"channel_shift": 0.08}, {"enhanced": False}, {"channel_quirk": False},
        {"sort_visible": False}, {"even_dissolve": False}, {"residue": 0.45},
    ])
    def test_the_setting_changes_the_middle_of_the_axis(self, recipe, sources, reglage):
        decay = Decay(frames=20)
        temoin = compose_at(recipe, decay, 160, 12, sources)
        modifie = compose_at(recipe, Decay(frames=20, **reglage), 160, 12, sources)
        assert not np.array_equal(temoin, modifie), f"{reglage} ne fait rien"

    def test_no_shift_leaves_the_channels_aligned(self, recipe, sources):
        """Un décalage nul ne doit pas désaligner par accident."""
        sans = compose_at(recipe, Decay(frames=20, channel_shift=0.0), 160, 12, sources)
        avec = compose_at(recipe, Decay(frames=20, channel_shift=0.06), 160, 12, sources)
        assert not np.array_equal(sans, avec)


class TestRobustness:
    def test_no_sources_is_refused(self):
        with pytest.raises(ValueError):
            compose_at(Recipe(sources=(), seed=1), Decay(), 128, 0, [])

    def test_an_empty_sequence_is_refused(self):
        with pytest.raises(ValueError):
            list(sequence(Recipe(sources=(), seed=1), Decay(), 128, []))

    def test_a_flat_image_still_decomposes(self):
        """Un aplat n'a pas de saillance : l'axe doit quand même avancer."""
        uni = [np.full((200, 200, 3), 180, np.uint8)] * 2
        recipe = Recipe(sources=("a", "b"), seed=5)
        decay = Decay(frames=10)
        assert not np.array_equal(
            compose_at(recipe, decay, 128, 0, uni),
            compose_at(recipe, decay, 128, 9, uni),
        )

    def test_a_two_frame_axis_works(self, recipe, sources):
        frames = list(sequence(recipe, Decay(frames=2), 96, sources))
        assert len(frames) == 2


class TestTheDrawnAxis:
    def test_a_drawn_axis_is_usable(self, recipe, sources):
        rng = random.Random(11)
        for _ in range(20):
            decay = random_decay(rng)
            assert 40 <= decay.frames <= 120
            assert decay.fps in (20, 24)
            assert 0.0 <= decay.stagger <= 1.0
            assert 0.0 <= decay.residue < 1.0
            assert decay.min_segment < decay.max_segment

    def test_a_drawn_axis_renders(self, recipe, sources):
        decay = random_decay(random.Random(13))
        out = compose_at(recipe, decay, 128, decay.frames // 2, sources)
        assert out.shape == (128, 128, 3)
        assert out.any(), "un axe tiré au hasard ne doit pas rendre du vide"
