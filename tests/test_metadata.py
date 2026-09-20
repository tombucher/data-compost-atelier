"""La recette voyage dans l'image, sans fichier supplémentaire.

Le projet parle de perdre des fichiers : un `.json` par image en doublerait le
nombre. Ces tests vérifient que l'aller-retour tient, et qu'une image sans
recette reste une image lisible.
"""
import numpy as np
import pytest
from PIL import Image

from atelier.engine import Effect, Recipe
from atelier.metadata import read_recipe, recipe_from_json, recipe_to_json, save_with_recipe


@pytest.fixture
def recipe():
    return Recipe(
        sources=("/un/chemin/fleur.jpg", "/un/chemin/pavot.jpg"),
        seed=987654,
        effects=(Effect("halftone", 0.012), Effect("saturate", 2.0)),
        saliency_threshold=140,
        smoothness=2.5,
        edge_blur=3,
        saturation_overflow=True,
    )


@pytest.fixture
def image():
    rng = np.random.default_rng(3)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


class TestRoundTrip:
    @pytest.mark.parametrize("suffix", [".png", ".jpg"])
    def test_recipe_survives_the_image(self, tmp_path, recipe, image, suffix):
        path = save_with_recipe(image, tmp_path / f"oeuvre{suffix}", recipe)
        back = read_recipe(path)

        assert back is not None
        assert back.seed == recipe.seed
        assert back.saliency_threshold == recipe.saliency_threshold
        assert back.smoothness == recipe.smoothness
        assert back.edge_blur == recipe.edge_blur
        assert back.saturation_overflow is recipe.saturation_overflow
        assert [(e.name, e.strength) for e in back.effects] == \
               [(e.name, e.strength) for e in recipe.effects]

    @pytest.mark.parametrize("suffix", [".png", ".jpg"])
    def test_no_companion_file_is_written(self, tmp_path, recipe, image, suffix):
        """Un seul fichier en sortie, c'est tout l'enjeu."""
        save_with_recipe(image, tmp_path / f"oeuvre{suffix}", recipe)
        assert len(list(tmp_path.iterdir())) == 1

    @pytest.mark.parametrize("suffix", [".png", ".jpg"])
    def test_the_image_stays_a_normal_image(self, tmp_path, recipe, image, suffix):
        path = save_with_recipe(image, tmp_path / f"oeuvre{suffix}", recipe)
        with Image.open(path) as opened:
            assert opened.size == (64, 64)

    def test_only_file_names_travel_not_paths(self, tmp_path, recipe, image):
        """Le dossier peut avoir bougé ; l'image reste exploitable."""
        path = save_with_recipe(image, tmp_path / "oeuvre.png", recipe)
        back = read_recipe(path)
        assert back.sources == ("fleur.jpg", "pavot.jpg")

    def test_sources_can_be_supplied_at_read_time(self, tmp_path, recipe, image):
        path = save_with_recipe(image, tmp_path / "oeuvre.png", recipe)
        back = read_recipe(path, sources=["/ailleurs/fleur.jpg"])
        assert back.sources == ("/ailleurs/fleur.jpg",)

    def test_extra_fields_are_carried(self, tmp_path, recipe, image):
        save_with_recipe(image, tmp_path / "o.png", recipe, rendered_at=1024)
        assert '"rendered_at":1024' in recipe_to_json(recipe, rendered_at=1024)


class TestMissingOrBroken:
    """Une image sans recette n'est pas une erreur."""

    def test_plain_image_returns_none(self, tmp_path, image):
        path = tmp_path / "venue_dailleurs.png"
        Image.fromarray(image).save(path)
        assert read_recipe(path) is None

    def test_absent_file_returns_none(self, tmp_path):
        assert read_recipe(tmp_path / "rien.png") is None

    def test_corrupt_metadata_returns_none(self, tmp_path, image):
        from PIL import PngImagePlugin
        info = PngImagePlugin.PngInfo()
        info.add_text("data-compost-atelier", "{ceci n'est pas du json")
        path = tmp_path / "abime.png"
        Image.fromarray(image).save(path, pnginfo=info)
        assert read_recipe(path) is None

    def test_a_non_image_returns_none(self, tmp_path):
        path = tmp_path / "pas_une_image.png"
        path.write_bytes(b"nullement une image")
        assert read_recipe(path) is None


class TestJsonShape:
    def test_defaults_fill_in_missing_fields(self):
        back = recipe_from_json('{"seed": 42}')
        assert back.seed == 42
        assert back.effects == ()
        assert back.saturation_overflow is True

    def test_json_is_compact(self, recipe):
        """La recette tient dans un champ de métadonnées, pas dans un fichier."""
        assert len(recipe_to_json(recipe)) < 400
