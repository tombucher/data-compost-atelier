"""Le tirage au hasard doit rester dans les habitudes du corpus.

Ces tests ne vérifient pas une image précise — le hasard en interdit l'idée.
Ils vérifient que la distribution des tirages ressemble à celle des œuvres
conservées, et qu'un tirage reste un point de départ exploitable : rejouable,
enregistrable, et jamais vide.
"""
import random
from collections import Counter

import numpy as np
import pytest

from atelier.chance import (
    FAMILIES,
    NEWCOMERS,
    PROFILES,
    WEIGHTS,
    draw_effects,
    random_decay,
    random_recipe,
)
from atelier.engine import EFFECTS, Recipe, render
from atelier.metadata import read_recipe, save_with_recipe


@pytest.fixture
def images():
    return [f"/dossier/image{i}.jpg" for i in range(12)]


class TestTheCorpusProportions:
    """Les parts mesurées sur les 230 œuvres doivent se retrouver au tirage."""

    def test_families_appear_in_corpus_proportion(self, images):
        rng = random.Random(7)
        dominante = Counter()
        for _ in range(4000):
            recipe = random_recipe(images, rng)
            # La matière de tête d'une famille est celle qui la nomme
            noms = {e.name for e in recipe.effects}
            for famille in FAMILIES:
                if PROFILES[famille]["always"][0] in noms:
                    dominante[famille] += 1

        # La trame domine largement : c'est le trait le plus net du corpus
        assert dominante["halftone"] > dominante["bitmap"] > dominante["saturate"]

    def test_the_drawn_weights_match_within_a_few_points(self, images):
        rng = random.Random(11)
        tirees = Counter(
            rng.choices(FAMILIES, weights=WEIGHTS, k=1)[0] for _ in range(20000)
        )
        attendu = dict(zip(FAMILIES, WEIGHTS))
        for famille, part in attendu.items():
            mesure = 100.0 * tirees[famille] / 20000
            assert abs(mesure - part) < 2.0, f"{famille} : {mesure:.1f} au lieu de {part}"

    def test_saturation_overflow_stays_the_exception(self, images):
        """Seize œuvres sur 230 en vivent : ce n'est pas la règle."""
        rng = random.Random(3)
        part = np.mean([random_recipe(images, rng).saturation_overflow
                        for _ in range(2000)])
        assert 0.10 < part < 0.22, f"débordement tiré {part:.0%} du temps"


class TestEachFamilyKeepsItsSignature:
    @pytest.mark.parametrize("famille", FAMILIES)
    def test_the_dominant_material_is_always_there(self, images, famille):
        rng = random.Random(13)
        attendu = PROFILES[famille]["always"][0]
        for _ in range(200):
            noms = [e.name for e in draw_effects(famille, rng)]
            assert attendu in noms

    @pytest.mark.parametrize("famille", FAMILIES)
    def test_strengths_stay_inside_the_measured_range(self, images, famille):
        rng = random.Random(17)
        nom, bas, haut = PROFILES[famille]["always"]
        for _ in range(200):
            force = next(e.strength for e in draw_effects(famille, rng)
                         if e.name == nom)
            assert bas <= force <= haut

    def test_bitmap_stays_near_full_strength(self, images):
        """En dessous de 0.85 le seuillage ne donne plus de noir et blanc."""
        assert PROFILES["bitmap"]["always"][1] >= 0.85

    @pytest.mark.parametrize("famille", FAMILIES)
    def test_the_family_can_be_asked_for(self, images, famille):
        recipe = random_recipe(images, random.Random(19), family=famille)
        assert PROFILES[famille]["always"][0] in {e.name for e in recipe.effects}


class TestTheDrawIsUsable:
    def test_every_drawn_effect_exists_in_the_engine(self, images):
        rng = random.Random(23)
        for _ in range(500):
            recipe = random_recipe(images, rng)
            groupes = [recipe.effects, *(g for g in recipe.per_image if g)]
            for groupe in groupes:
                for effect in groupe:
                    assert effect.name in EFFECTS, effect.name

    def test_sources_are_drawn_without_repetition(self, images):
        rng = random.Random(29)
        for _ in range(300):
            sources = random_recipe(images, rng).sources
            assert 2 <= len(sources) <= 4
            assert len(set(sources)) == len(sources)

    def test_a_single_available_image_still_works(self):
        recipe = random_recipe(["/seule.jpg"], random.Random(31))
        assert recipe.sources == ("/seule.jpg",)

    def test_no_images_is_refused(self):
        with pytest.raises(ValueError):
            random_recipe([], random.Random(37))

    def test_threshold_stays_a_percentile(self, images):
        rng = random.Random(41)
        for _ in range(300):
            assert 0 <= random_recipe(images, rng).saliency_threshold <= 100

    def test_the_draw_is_replayable(self, images):
        """Même rng, même tirage : sinon on ne pourrait pas y revenir."""
        a = random_recipe(images, random.Random(43))
        b = random_recipe(images, random.Random(43))
        assert a == b

    def test_two_draws_differ(self, images):
        rng = random.Random(47)
        tirages = {random_recipe(images, rng) for _ in range(50)}
        assert len(tirages) > 45, "le tirage doit rester du hasard"


class TestItSurvivesTheRoundTrip:
    """Un tirage doit pouvoir se reprendre comme n'importe quel export."""

    def test_a_drawn_recipe_goes_through_the_metadata(self, tmp_path):
        recipe = random_recipe(["/x/a.jpg", "/x/b.jpg", "/x/c.jpg"], random.Random(53))
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        back = read_recipe(save_with_recipe(image, tmp_path / "o.png", recipe))

        assert back.seed == recipe.seed
        assert back.saliency_threshold == recipe.saliency_threshold
        assert back.saturation_overflow is recipe.saturation_overflow
        assert [(e.name, round(e.strength, 6)) for e in back.effects] == \
               [(e.name, round(e.strength, 6)) for e in recipe.effects]
        assert len(back.per_image) == len(recipe.per_image)


class TestItActuallyRenders:
    """Un tirage qui ne compose rien ne servirait à rien."""

    @pytest.fixture
    def sources(self):
        rng = np.random.default_rng(5)
        return [rng.integers(0, 255, (200, 150, 3), dtype=np.uint8) for _ in range(6)]

    @pytest.mark.parametrize("famille", FAMILIES)
    def test_each_family_produces_an_image(self, sources, famille):
        noms = [f"i{i}.jpg" for i in range(6)]
        recipe = random_recipe(noms, random.Random(59), family=famille)
        choisies = [sources[noms.index(s)] for s in recipe.sources]
        out = render(recipe, size=160, sources=choisies)

        assert out.shape == (160, 160, 3)
        assert out.any(), "la toile ne doit pas rester vide"
        assert out.std() > 5, "un tirage plat ne serait pas une composition"

    def test_the_effects_actually_bite(self, sources):
        """Sans matières, le rendu diffère : le tirage ne tourne pas à vide."""
        noms = [f"i{i}.jpg" for i in range(6)]
        recipe = random_recipe(noms, random.Random(61))
        choisies = [sources[noms.index(s)] for s in recipe.sources]
        nu = Recipe(sources=recipe.sources, seed=recipe.seed,
                    saliency_threshold=recipe.saliency_threshold,
                    smoothness=recipe.smoothness, edge_blur=recipe.edge_blur)
        assert not np.array_equal(
            render(recipe, size=160, sources=choisies),
            render(nu, size=160, sources=choisies),
        )


class TestTheAxisDraw:
    def test_a_drawn_axis_lasts_a_plausible_time(self):
        rng = random.Random(67)
        for _ in range(200):
            decay = random_decay(rng)
            assert 40 <= decay.frames <= 120
            assert decay.fps in (20, 24)
            assert decay.min_segment < decay.max_segment
            # Aux extrêmes, le décalage écrase le propos : tout pourrit
            # ensemble, ou la pile devient un diaporama.
            assert 0.2 < decay.stagger < 0.7
