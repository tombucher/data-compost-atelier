"""Jusqu'où les matières portent, et sous quel angle.

Deux constats sont à l'origine de ce fichier. D'abord les matières mordaient
sur la zone la moins saillante, qui ne porte qu'une petite part du poids de
la composition : tout ce qui réordonne sans transformer — un tri, un décalage
— s'y noyait, au point de paraître sans effet. Ensuite elles travaillaient
toutes par lignes horizontales, sans moyen de les incliner.

Ces tests mesurent donc des amplitudes, pas seulement des différences : un
effet qui change trois pixels sur mille est indistinguable d'un effet absent,
et c'est précisément le défaut qu'on veut empêcher de revenir.
"""
import glob

import cv2
import numpy as np
import pytest

from atelier.engine import (
    Effect,
    Recipe,
    layer_geometry,
    render,
    saliency_of,
    smooth_mask,
)
from atelier.metadata import read_recipe, save_with_recipe

# Les matières qui réordonnent sans changer les valeurs : ce sont elles qui
# se noyaient. On les éprouve ensemble.
DISCRETES = (("pixelsort", 0.04), ("mosh", 0.03))


@pytest.fixture(scope="module")
def photos():
    chemins = sorted(glob.glob("compost-test/*.jpg"))[:3]
    if len(chemins) < 2:
        pytest.skip("il faut des photographies pour mesurer une amplitude")
    return chemins, [cv2.imread(p) for p in chemins]


def ecart(a, b):
    """L'amplitude du changement, en niveaux."""
    return float(np.abs(a.astype(int) - b.astype(int)).mean())


class TestTheQuietZoneCarriesLittleWeight:
    """Le constat d'où vient la bavure, mesuré plutôt qu'affirmé."""

    def test_the_attacked_zone_is_the_lightest(self, photos):
        _, images = photos
        carte = smooth_mask(saliency_of(images[0]), 3.0, 0)
        poids = carte.astype(float) / 255.0
        calme = carte < np.quantile(carte, 0.60)
        part = poids[calme].sum() / poids.sum()
        assert part < 0.30, (
            "si la zone calme portait le poids de la composition, la bavure "
            f"n'aurait pas lieu d'être ; elle en porte {part:.0%}"
        )


class TestBleedMakesMaterialsVisible:
    @pytest.mark.parametrize("nom,force", DISCRETES)
    def test_bleeding_widens_the_reach(self, photos, nom, force):
        chemins, images = photos
        nu = render(Recipe(sources=tuple(chemins), seed=7), size=500, sources=images)

        def rendu(bave):
            return render(
                Recipe(sources=tuple(chemins), seed=7, bleed=bave,
                       effects=(Effect(nom, force),)),
                size=500, sources=images,
            )

        serre = ecart(rendu(0.0), nu)
        large = ecart(rendu(1.0), nu)
        assert large > serre * 2.5, \
            f"{nom} : la bavure doit élargir nettement ({serre:.2f} → {large:.2f})"

    @pytest.mark.parametrize("nom,force", DISCRETES)
    def test_it_is_visible_at_full_bleed(self, photos, nom, force):
        """Le seuil est bas exprès : on garde contre l'effet imperceptible."""
        chemins, images = photos
        nu = render(Recipe(sources=tuple(chemins), seed=7), size=500, sources=images)
        plein = render(
            Recipe(sources=tuple(chemins), seed=7, bleed=1.0,
                   effects=(Effect(nom, force),)),
            size=500, sources=images,
        )
        assert ecart(plein, nu) > 8.0, f"{nom} reste invisible même à pleine bavure"

    def test_no_bleed_changes_nothing_by_itself(self, photos):
        chemins, images = photos
        sans = Recipe(sources=tuple(chemins), seed=7)
        avec = Recipe(sources=tuple(chemins), seed=7, bleed=0.8)
        assert np.array_equal(
            render(sans, size=300, sources=images),
            render(avec, size=300, sources=images),
        ), "sans matière, la bavure n'a rien à élargir"

    def test_bleed_is_clamped(self, photos):
        chemins, images = photos
        for bave in (-2.0, 5.0):
            out = render(
                Recipe(sources=tuple(chemins), seed=7, bleed=bave,
                       effects=(Effect("halftone", 0.015),)),
                size=250, sources=images)
            assert out.shape == (250, 250, 3)


class TestMoshingIsActuallyVisible:
    """Le tri par canaux était annulé par la bizarrerie du rouge."""

    def test_it_never_does_nothing(self, photos):
        """Une graine sur quatre ne triait rien du tout."""
        chemins, images = photos
        nu = render(Recipe(sources=tuple(chemins), seed=1), size=400, sources=images)
        for graine in range(1, 9):
            out = render(
                Recipe(sources=tuple(chemins), seed=graine, bleed=1.0,
                       effects=(Effect("mosh", 0.03),)),
                size=400, sources=images)
            temoin = render(Recipe(sources=tuple(chemins), seed=graine),
                            size=400, sources=images)
            assert ecart(out, temoin) > 5.0, f"graine {graine} : le mosh ne fait rien"

    def test_it_separates_the_channels(self, photos):
        chemins, images = photos

        def separation(image):
            b, g, r = cv2.split(image.astype(int))
            return (np.abs(b - g).mean() + np.abs(g - r).mean()) / 2

        nu = render(Recipe(sources=tuple(chemins), seed=7), size=400, sources=images)
        moshe = render(
            Recipe(sources=tuple(chemins), seed=7, bleed=1.0,
                   effects=(Effect("mosh", 0.03),)),
            size=400, sources=images)
        assert separation(moshe) > separation(nu) * 1.2, \
            "le datamoshing doit décoller les canaux les uns des autres"


class TestTheAngleTurnsTheMaterials:
    TOURNANTES = (("halftone", 0.016, "line"), ("halftone", 0.016, "round"),
                  ("pixelsort", 0.04, "round"), ("mosh", 0.03, "round"),
                  ("shift", 0.01, "round"))

    @pytest.mark.parametrize("nom,force,forme", TOURNANTES)
    def test_turning_changes_the_image(self, photos, nom, force, forme):
        chemins, images = photos

        def rendu(angle):
            return render(
                Recipe(sources=tuple(chemins), seed=7, bleed=0.8,
                       effects=(Effect(nom, force, forme, angle),)),
                size=400, sources=images)

        droit = rendu(0)
        for angle in (45, 90):
            assert ecart(rendu(angle), droit) > 1.0, \
                f"{nom} {forme} ne tourne pas à {angle}°"

    def test_a_line_screen_turns_the_most(self, photos):
        """C'est la trame ligne qui rend l'angle le plus lisible."""
        chemins, images = photos

        def rendu(angle, forme):
            return render(
                Recipe(sources=tuple(chemins), seed=7, bleed=0.8,
                       effects=(Effect("halftone", 0.016, forme, angle),)),
                size=400, sources=images)

        ligne = ecart(rendu(90, "line"), rendu(0, "line"))
        assert ligne > 10.0, "une trame ligne tournée à 90° doit changer du tout au tout"

    def test_zero_and_a_full_turn_agree(self, photos):
        """180° ramène les lignes sur le même axe."""
        chemins, images = photos

        def rendu(angle):
            return render(
                Recipe(sources=tuple(chemins), seed=7, bleed=0.5,
                       effects=(Effect("halftone", 0.016, "line", angle),)),
                size=300, sources=images)

        assert np.array_equal(rendu(0), rendu(180))

    def test_an_angle_on_a_still_material_changes_nothing(self, photos):
        """Le bitmap ne travaille pas par lignes : son axe ne doit rien faire."""
        chemins, images = photos
        droit = Recipe(sources=tuple(chemins), seed=7,
                       effects=(Effect("bitmap", 0.8),))
        tourne = Recipe(sources=tuple(chemins), seed=7,
                        effects=(Effect("bitmap", 0.8, "round", 45),))
        assert np.array_equal(
            render(droit, size=300, sources=images),
            render(tourne, size=300, sources=images),
        )

    def test_each_material_keeps_its_own_axis(self, photos):
        """Le point de tout : une trame inclinée sur un tri à l'horizontale."""
        chemins, images = photos

        def rendu(axe_trame, axe_mosh):
            return render(
                Recipe(sources=tuple(chemins), seed=7, bleed=0.8, effects=(
                    Effect("mosh", 0.03, "round", axe_mosh),
                    Effect("halftone", 0.016, "line", axe_trame))),
                size=400, sources=images)

        assert not np.array_equal(rendu(45, 0), rendu(45, 45)), \
            "changer le seul axe du tri doit changer l'image"
        assert not np.array_equal(rendu(0, 45), rendu(45, 45)), \
            "changer le seul axe de la trame doit changer l'image"


class TestFraming:
    def test_full_frame_leaves_no_background(self, photos):
        chemins, images = photos
        pose = render(Recipe(sources=tuple(chemins), seed=7), size=400, sources=images)
        plein = render(Recipe(sources=tuple(chemins), seed=7, full_frame=True),
                       size=400, sources=images)
        assert np.mean(pose.max(axis=2) == 0) > 0.05, "le fond doit se voir, posé"
        assert np.mean(plein.max(axis=2) == 0) < 0.01, "plein cadre, il ne doit rester aucun fond"

    def test_the_image_covers_the_canvas(self):
        """Plein cadre, le plus petit côté touche les bords."""
        from atelier.engine import Placement
        image = np.zeros((300, 200, 3), np.uint8)
        target, x, y = layer_geometry(image, Placement(0.5, 0.5), 400, True)
        assert min(target) >= 400
        assert x <= 0 and y <= 0, "l'image déborde, donc se recadre"

    def test_posed_framing_stays_inside(self):
        from atelier.engine import Placement
        image = np.zeros((300, 200, 3), np.uint8)
        target, x, y = layer_geometry(image, Placement(0.5, 0.5), 400, False)
        assert max(target) <= 400
        assert x >= 0 and y >= 0

    @pytest.mark.parametrize("size", [128, 400, 700])
    def test_both_framings_fill_the_asked_size(self, photos, size):
        chemins, images = photos
        for plein in (False, True):
            out = render(Recipe(sources=tuple(chemins), seed=7, full_frame=plein),
                         size=size, sources=images)
            assert out.shape == (size, size, 3)

    def test_framing_survives_the_axis(self, photos):
        from atelier.decay import Decay, compose_at
        chemins, images = photos
        recipe = Recipe(sources=tuple(chemins), seed=7, full_frame=True)
        out = compose_at(recipe, Decay(frames=12), 300, 6, images)
        assert out.shape == (300, 300, 3)
        assert np.mean(out.max(axis=2) == 0) < 0.35, \
            "plein cadre, l'axe ne doit pas rouvrir un fond noir avant la fin"


class TestEdgesOnly:
    """À 1, le réglage effaçait toute l'image."""

    @pytest.mark.parametrize("bords", [1, 2, 3, 5, 9, 15])
    def test_a_ring_survives_at_every_setting(self, photos, bords):
        _, images = photos
        carte = smooth_mask(saliency_of(images[0]), 3.0, bords)
        assert carte.max() > 100, f"à {bords}, il ne reste pas de liseré"
        assert carte.mean() < 200, f"à {bords}, ce n'est plus un liseré mais un aplat"

    def test_the_image_does_not_vanish(self, photos):
        chemins, images = photos
        out = render(Recipe(sources=tuple(chemins), seed=7, edge_blur=1),
                     size=300, sources=images)
        assert out.any(), "l'image entière disparaissait avec un noyau de 1"

    def test_zero_leaves_the_map_alone(self, photos):
        _, images = photos
        carte = saliency_of(images[0])
        assert np.array_equal(smooth_mask(carte.copy(), 0, 0), carte)


class TestTheSettingsTravel:
    def test_they_survive_the_metadata(self, tmp_path):
        recipe = Recipe(sources=("a.jpg",), seed=8, bleed=0.65, full_frame=True,
                        effects=(Effect("mosh", 0.03, "round", 37.5),
                                 Effect("halftone", 0.01, "line", 120.0)))
        back = read_recipe(save_with_recipe(
            np.zeros((32, 32, 3), np.uint8), tmp_path / "o.png", recipe))
        assert back.bleed == pytest.approx(0.65)
        assert back.full_frame is True
        assert [e.angle for e in back.effects] == [37.5, 120.0], \
            "chaque matière doit retrouver son propre axe"

    def test_older_images_keep_the_old_behaviour(self):
        from atelier.metadata import recipe_from_json
        back = recipe_from_json(
            '{"seed": 1, "effects": [{"name": "mosh", "strength": 0.03}]}')
        assert back.bleed == 0.0
        assert back.full_frame is False
        assert back.effects[0].angle == 0.0

    def test_a_flat_angle_is_not_written_out(self):
        from atelier.metadata import recipe_to_json
        recipe = Recipe(sources=("a",), seed=1, effects=(Effect("mosh", 0.03),))
        assert '"angle"' not in recipe_to_json(recipe)
