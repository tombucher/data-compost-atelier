"""Le tri vectorisé doit rendre exactement ce que rendaient les boucles.

Comme pour la trame, l'implémentation d'origine est recopiée ici et sert de
juge. Sans elle, « vectorisé » serait une promesse.

Ce fichier ne couvre que les gestes : trier, décaler, dissoudre. Le temps qui
les enchaîne est éprouvé dans `test_decay.py`.
"""
import random

import cv2
import numpy as np
import pytest

from atelier.video import mask_sequence, pixel_sort, shift_channels, write_video


# --- implémentation d'origine, recopiée de datamoshing4.py -----------------
# Le tirage aléatoire d'inversion est retiré : il rendait toute comparaison
# impossible, et c'est précisément ce que la graine remplace.

def original_pixel_sort(image, mask, sort_method="brightness", vertical=False,
                        min_length=20, max_length=100):
    result = image.copy()
    height, width = image.shape[:2]
    mask_binary = (mask.astype(np.float32) > 0.5).astype(np.uint8)
    if sort_method in ("hue", "saturation"):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def runs(line, length):
        found, start = [], None
        for i in range(length):
            if line[i] and start is None:
                start = i
            elif (not line[i] or i == length - 1) and start is not None:
                end = i if not line[i] else i + 1
                if end - start > max_length:
                    for sub in range(start, end, max_length):
                        stop = min(sub + max_length, end)
                        if stop - sub >= min_length:
                            found.append((sub, stop))
                elif end - start >= min_length:
                    found.append((start, end))
                start = None
        return found

    if vertical:
        for x in range(width):
            for start, end in runs(mask_binary[:, x], height):
                segment = result[start:end, x].copy()
                if sort_method == "hue":
                    values = hsv[start:end, x, 0]
                elif sort_method == "saturation":
                    values = hsv[start:end, x, 1]
                else:
                    values = np.sum(segment, axis=1)
                result[start:end, x] = segment[np.argsort(values, kind="stable")]
    else:
        for y in range(height):
            for start, end in runs(mask_binary[y, :], width):
                segment = result[y, start:end].copy()
                if sort_method == "hue":
                    values = hsv[y, start:end, 0]
                elif sort_method == "saturation":
                    values = hsv[y, start:end, 1]
                else:
                    values = np.sum(segment, axis=1)
                result[y, start:end] = segment[np.argsort(values, kind="stable")]
    return result


@pytest.fixture
def photo():
    rng = np.random.default_rng(11)
    blobs = rng.integers(0, 255, (12, 12, 3), dtype=np.uint8)
    return cv2.resize(blobs, (192, 192), interpolation=cv2.INTER_CUBIC)


@pytest.fixture
def patchy():
    """Un masque à plages de longueurs variées, pour éprouver le découpage."""
    mask = np.zeros((192, 192), dtype=bool)
    mask[10:40, :] = True      # plage moyenne
    mask[60:180, :] = True     # plage longue, à découper
    mask[:, 100:105] = False   # trous verticaux
    mask[185:188, :] = True    # plage trop courte, à ne pas trier
    return mask


class TestPixelSortMatchesTheOriginal:
    @pytest.mark.parametrize("vertical", [True, False])
    @pytest.mark.parametrize("method", ["brightness", "hue", "saturation"])
    def test_identical_output(self, photo, patchy, vertical, method):
        mine = pixel_sort(photo, patchy, method, vertical, 20, 60)
        theirs = original_pixel_sort(photo, patchy, method, vertical, 20, 60)
        assert np.array_equal(mine, theirs)

    @pytest.mark.parametrize("min_len,max_len", [(5, 20), (20, 60), (30, 250)])
    def test_segment_lengths_agree(self, photo, patchy, min_len, max_len):
        mine = pixel_sort(photo, patchy, "brightness", True, min_len, max_len)
        theirs = original_pixel_sort(photo, patchy, "brightness", True, min_len, max_len)
        assert np.array_equal(mine, theirs)

    def test_full_mask(self, photo):
        full = np.ones(photo.shape[:2], bool)
        assert np.array_equal(
            pixel_sort(photo, full, "brightness", True, 20, 60),
            original_pixel_sort(photo, full, "brightness", True, 20, 60),
        )

    def test_empty_mask_changes_nothing(self, photo):
        empty = np.zeros(photo.shape[:2], bool)
        assert np.array_equal(pixel_sort(photo, empty, "brightness", True, 20, 60), photo)

    def test_short_runs_are_left_alone(self, photo):
        """Une plage plus courte que le minimum ne doit pas bouger."""
        mask = np.zeros(photo.shape[:2], bool)
        mask[50:55, :] = True                     # 5 pixels, minimum 20
        out = pixel_sort(photo, mask, "brightness", True, 20, 60)
        assert np.array_equal(out, photo)


class TestSpeed:
    def test_fast_enough_for_print_size(self, ):
        """En 2000 px, la version d'origine demanderait des minutes par frame."""
        import time
        rng = np.random.default_rng(2)
        image = rng.integers(0, 255, (2000, 2000, 3), dtype=np.uint8)
        mask = np.ones((2000, 2000), bool)
        start = time.perf_counter()
        pixel_sort(image, mask, "brightness", True, 60, 200)
        assert time.perf_counter() - start < 15.0


class TestSeeded:
    """Une séquence doit pouvoir se rejouer."""

    def test_flips_are_reproducible(self, photo, patchy):
        a = pixel_sort(photo, patchy, "brightness", True, 20, 60,
                       np.random.default_rng(5))
        b = pixel_sort(photo, patchy, "brightness", True, 20, 60,
                       np.random.default_rng(5))
        assert np.array_equal(a, b)

    def test_flips_change_the_result(self, photo, patchy):
        plain = pixel_sort(photo, patchy, "brightness", True, 20, 60)
        flipped = pixel_sort(photo, patchy, "brightness", True, 20, 60,
                             np.random.default_rng(5))
        assert not np.array_equal(plain, flipped)

    def test_channel_shift_follows_the_draw(self, photo):
        a = shift_channels(photo, 8, random.Random(3))
        b = shift_channels(photo, 8, random.Random(3))
        assert np.array_equal(a, b)

    def test_no_shift_is_a_no_op(self, photo):
        assert np.array_equal(shift_channels(photo, 0, random.Random(1)), photo)


class TestWriting:
    def test_writes_a_playable_file(self, tmp_path):
        frames = [np.full((64, 64, 3), v, np.uint8) for v in (20, 90, 160, 230)]
        target = tmp_path / "essai.mp4"
        assert write_video(iter(frames), target, fps=8) == 4
        assert target.is_file() and target.stat().st_size > 0

        back = cv2.VideoCapture(str(target))
        assert back.isOpened()
        assert int(back.get(cv2.CAP_PROP_FRAME_COUNT)) >= 3
        back.release()

    def test_empty_sequence_writes_nothing(self, tmp_path):
        assert write_video(iter([]), tmp_path / "vide.mp4", fps=8) == 0


class TestMasks:
    def test_sequence_opens_and_closes(self):
        field = np.linspace(0, 255, 64 * 64).reshape(64, 64).astype(np.uint8)
        masks = mask_sequence(field, 5)
        assert len(masks) == 5
        assert masks[0].mean() > masks[-1].mean(), "le masque doit se refermer"


class TestDissolve:
    """La dissolution doit occuper toute la séquence, pas deux images."""

    @pytest.fixture
    def concentrated(self):
        """Une carte comme en produit le résidu spectral.

        Distribution exponentielle : l'essentiel près de zéro, une traîne
        jusqu'au maximum. Mesuré sur les photographies du projet, médiane
        0,02 et moyenne 0,04 pour un maximum à 1.
        """
        rng = np.random.default_rng(3)
        field = rng.exponential(0.04, (128, 128))
        field[60:70, 60:70] = 1.0
        return np.clip(field * 255, 0, 255).astype(np.uint8)

    def test_even_dissolve_spreads_over_the_sequence(self, concentrated):
        parts = [m.mean() for m in mask_sequence(concentrated, 12, even=True)]
        # La part conservée doit décroître régulièrement, pas s'effondrer
        assert parts[0] > 0.9 and parts[-1] < 0.1
        milieu = parts[len(parts) // 2]
        assert 0.25 < milieu < 0.75, f"à mi-course il reste {milieu:.0%}"

    def test_raw_thresholds_collapse_immediately(self, concentrated):
        """Le comportement d'origine, conservé pour comparaison."""
        parts = [m.mean() for m in mask_sequence(concentrated, 12, even=False)]
        assert parts[2] < 0.05, "l'original vidait l'image en deux images"

    def test_a_flat_map_still_dissolves(self):
        """Sans relief, les quantiles se confondraient et rien ne bougerait.

        Une image unie doit quand même laisser la place à la suivante : on
        se rabat sur un balayage spatial.
        """
        flat = np.full((64, 64), 30, np.uint8)
        parts = [m.mean() for m in mask_sequence(flat, 10, even=True)]
        assert parts[0] == 1.0, "tout est là au départ"
        assert parts[-1] == 0.0, "la seconde image doit finir par tout occuper"
        assert 0.2 < parts[5] < 0.9, f"la transition doit être progressive ({parts[5]:.0%})"

    def test_the_last_mask_is_always_empty(self, concentrated):
        """Sans quoi la seconde image n'arriverait jamais tout à fait."""
        for frames in (4, 12, 30):
            assert mask_sequence(concentrated, frames)[-1].max() == 0

    def test_masks_only_shrink(self, concentrated):
        parts = [m.mean() for m in mask_sequence(concentrated, 16, even=True)]
        assert all(b <= a + 1e-6 for a, b in zip(parts, parts[1:]))


