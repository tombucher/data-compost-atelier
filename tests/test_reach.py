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


class TestTheMoshDirection:
    """Le vert partait perpendiculairement aux deux autres, sans qu'on l'ait voulu.

    Ce n'est pas une invention : l'original envoie bien le vert à
    contre-sens. Mais sa bizarrerie laissait le rouge intact et le bleu trié
    une fois sur deux, si bien que le vert était souvent le seul canal
    réellement trié — on ne voyait donc qu'une direction. Rendre les trois
    canaux actifs a rendu la croix visible d'un coup.
    """

    @staticmethod
    def _sens(image):
        """Écart entre variation horizontale et verticale : plus il est
        grand, plus les coulées vont dans un seul sens."""
        gris = image.astype(float)
        horiz = float(np.abs(np.diff(gris, axis=1)).mean())
        verti = float(np.abs(np.diff(gris, axis=0)).mean())
        return abs(horiz - verti)

    def _rendu(self, photos, croise):
        chemins, images = photos
        return render(
            Recipe(sources=tuple(chemins), seed=7, bleed=1.0,
                   effects=(Effect("mosh", 0.03, "round", 0.0, croise),)),
            size=400, sources=images)

    def test_uncrossed_keeps_a_single_direction(self, photos):
        droit = self._sens(self._rendu(photos, False))
        croise = self._sens(self._rendu(photos, True))
        assert droit > croise * 1.5, (
            "décroisé, les coulées doivent aller dans un seul sens "
            f"(asymétrie {droit:.1f} contre {croise:.1f})"
        )

    def test_it_is_uncrossed_by_default(self):
        assert Effect("mosh", 0.03).cross is False

    def test_crossing_changes_the_image(self, photos):
        assert not np.array_equal(self._rendu(photos, False),
                                  self._rendu(photos, True))

    def test_the_axis_keeps_the_original_behaviour(self):
        """Dans l'axe du temps, le vert reste à contre-sens comme en 2025."""
        import inspect

        from atelier.video import sort_channels_separately
        defaut = inspect.signature(sort_channels_separately).parameters["cross"].default
        assert defaut is True

    def test_it_survives_the_metadata(self, tmp_path):
        recipe = Recipe(sources=("a.jpg",), seed=3,
                        effects=(Effect("mosh", 0.03, "round", 0.0, True),))
        back = read_recipe(save_with_recipe(
            np.zeros((32, 32, 3), np.uint8), tmp_path / "o.png", recipe))
        assert back.effects[0].cross is True


class TestTheBleedActuallyBleeds:
    """Une bavure qui déplace la coupure sans la fondre n'est pas une bavure.

    Élargir la zone traitée ne suffit pas : un masque booléen tranche au
    rasoir où qu'on le place, et la normalisation du poids rend sa pleine
    intensité au moindre pixel couvert. Le bord de la matière restait donc
    une découpe nette sur le fond, quelle que soit la bavure.
    """

    @staticmethod
    def _pente_au_bord(image):
        """La raideur de la frontière entre la matière et le vide.

        Mesurée sur le seul pourtour des plages noires : le grain interne du
        datamoshing n'a rien à voir avec la question.
        """
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vide = (gris < 10).astype(np.uint8)
        noyau = np.ones((7, 7), np.uint8)
        pourtour = cv2.dilate(vide, noyau) - cv2.erode(vide, noyau)
        if pourtour.sum() == 0:
            return 0.0
        pente = cv2.Sobel(gris.astype(np.float32), cv2.CV_32F, 1, 1, ksize=5)
        return float(np.abs(pente)[pourtour > 0].mean())

    def _vue(self, photos, bave):
        from atelier.decay import Decay, compose_at
        chemins, images = photos
        recipe = Recipe(sources=tuple(chemins), seed=7, bleed=bave,
                        effects=(Effect("mosh", 0.05),))
        return compose_at(recipe, Decay(frames=24), 460, 12, images)

    def test_the_edge_softens(self, photos):
        franc = self._pente_au_bord(self._vue(photos, 0.0))
        fondu = self._pente_au_bord(self._vue(photos, 0.6))
        assert fondu < franc * 0.6, (
            f"la coupure doit se fondre, pas seulement se déplacer "
            f"({franc:.0f} → {fondu:.0f})"
        )

    def test_less_of_the_image_falls_to_pure_black(self, photos):
        noir = lambda v: float(np.mean(v.max(axis=2) == 0))
        assert noir(self._vue(photos, 0.8)) < noir(self._vue(photos, 0.0))

    @pytest.mark.parametrize("bave", [0.0, 0.5, 1.0])
    def test_the_start_is_never_touched(self, photos, bave):
        """Le fondu ne mord qu'une fois l'axe engagé."""
        from atelier.decay import Decay, compose_at
        chemins, images = photos
        recipe = Recipe(sources=tuple(chemins), seed=7, bleed=bave,
                        effects=(Effect("mosh", 0.04),))
        assert np.array_equal(
            render(recipe, size=300, sources=images),
            compose_at(recipe, Decay(), 300, 0, images),
        )

    def test_no_bleed_leaves_the_edge_alone(self, photos):
        """Sans bavure, la découpe reste franche : c'est le rendu d'origine."""
        from atelier.decay import Decay, compose_at
        chemins, images = photos
        recipe = Recipe(sources=tuple(chemins), seed=7, effects=(Effect("mosh", 0.05),))
        deux = [compose_at(recipe, Decay(frames=24), 300, 12, images) for _ in range(2)]
        assert np.array_equal(*deux)
        assert self._pente_au_bord(deux[0]) > 80, \
            "à bavure nulle, le bord doit rester une coupure"


class TestTheOutputFrame:
    """Un cadre n'est pas un rognage.

    Rogner une image déjà rendue perdrait la définition qu'on vient de
    demander. Le cadre dit quelle part de la toile on garde, et la toile est
    alors calculée d'autant plus grande : la part gardée sort à la taille
    voulue, avec tout son détail.
    """

    def test_the_canvas_grows_with_a_tighter_frame(self):
        from atelier.engine import toile_pour
        moitie = Recipe(sources=("a",), seed=1, crop=(0.25, 0.25, 0.5))
        assert toile_pour(moitie, 4000)[0] == 8000
        assert toile_pour(Recipe(sources=("a",), seed=1), 4000)[0] == 4000

    def test_the_canvas_is_capped(self):
        """Seize gigaoctets de toile, c'est une machine qui ne rend plus la main.

        Une toile porte seize octets par pixel — un canvas en float32 à
        trois canaux plus sa carte de poids. Un cadre serré sur un grand
        tirage demandait 32000 px de côté, soit 16 Go, sans prévenir.
        """
        from atelier.engine import TOILE_MAX, toile_pour
        serre = Recipe(sources=("a",), seed=1, crop=(0.4, 0.4, 0.25))
        toile, _, sortie = toile_pour(serre, 8000)
        assert toile == TOILE_MAX
        assert sortie < 8000, "une sortie bridée doit s'annoncer plus petite"
        assert toile * toile * 16 < 2e9, "la toile doit rester tenable"

    def test_an_unbridled_frame_keeps_the_asked_size(self):
        from atelier.engine import toile_pour
        large = Recipe(sources=("a",), seed=1, crop=(0.2, 0.2, 0.6))
        toile, _, sortie = toile_pour(large, 4000)
        assert sortie == 4000, "sous le plafond, la taille demandée est tenue"

    def test_the_frame_comes_out_at_the_asked_size(self, photos):
        from atelier.engine import decouper, toile_pour
        chemins, images = photos
        recipe = Recipe(sources=tuple(chemins), seed=9, crop=(0.2, 0.2, 0.4),
                        effects=(Effect("halftone", 0.012),))
        toile, cadre, taille = toile_pour(recipe, 500)
        out = decouper(render(recipe, size=toile, sources=images), cadre, taille)
        assert out.shape == (500, 500, 3)

    def test_the_frame_shows_the_right_part(self, photos):
        """La zone tirée doit être celle qu'on a désignée, pas une autre."""
        from atelier.engine import decouper, toile_pour
        chemins, images = photos
        nu = Recipe(sources=tuple(chemins), seed=9, effects=(Effect("halftone", 0.012),))
        entier = render(nu, size=600, sources=images)

        recipe = Recipe(sources=tuple(chemins), seed=9, crop=(0.25, 0.25, 0.4),
                        effects=(Effect("halftone", 0.012),))
        toile, cadre, taille = toile_pour(recipe, 600)
        sortie = decouper(render(recipe, size=toile, sources=images), cadre, taille)

        # La même zone, prise dans le rendu entier puis agrandie
        temoin = cv2.resize(entier[150:150 + 240, 150:150 + 240], (600, 600))
        ailleurs = cv2.resize(entier[0:240, 0:240], (600, 600))
        assert ecart(sortie, temoin) < ecart(sortie, ailleurs), \
            "le cadre doit livrer la zone désignée"

    @pytest.mark.parametrize("brut,attendu", [
        ((0.0, 0.0, 1.0), None),          # couvre tout : pas un cadre
        ((0.0, 0.0, 2.0), None),          # débordement ramené à tout
        ((0.9, 0.9, 0.5), (0.5, 0.5, 0.5)),   # ramené dans la toile
        ((-1.0, -1.0, 0.3), (0.0, 0.0, 0.3)),
    ])
    def test_a_frame_stays_inside_the_canvas(self, brut, attendu):
        assert Recipe(sources=("a",), seed=1, crop=brut).crop_clair() == attendu

    def test_no_frame_changes_nothing(self, photos):
        chemins, images = photos
        recipe = Recipe(sources=tuple(chemins), seed=9,
                        effects=(Effect("halftone", 0.012),))
        from atelier.engine import decouper, toile_pour
        toile, cadre, taille = toile_pour(recipe, 400)
        assert cadre is None and taille == 400
        assert np.array_equal(
            decouper(render(recipe, size=toile, sources=images), cadre, taille),
            render(recipe, size=400, sources=images),
        )

    def test_it_survives_the_metadata(self, tmp_path):
        recipe = Recipe(sources=("a.jpg",), seed=4, crop=(0.1, 0.2, 0.55))
        back = read_recipe(save_with_recipe(
            np.zeros((32, 32, 3), np.uint8), tmp_path / "o.png", recipe))
        assert back.crop == (0.1, 0.2, 0.55)

    def test_older_images_have_no_frame(self):
        from atelier.metadata import recipe_from_json
        assert recipe_from_json('{"seed": 1}').crop is None
