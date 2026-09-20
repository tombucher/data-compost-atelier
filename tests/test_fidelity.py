"""Le rendu doit rester celui des œuvres de 2024.

Ces tests comparent le moteur à l'implémentation d'origine, recopiée ici
telle quelle. Sans eux, « vectoriser le halftone » ou « expliciter le
débordement » resteraient des affirmations.
"""
import cv2
import numpy as np
import pytest

from atelier.engine import apply_halftone, apply_saturate


# --- implémentations d'origine, recopiées de blending-image2.py -------------

def original_halftone(image, mask, strength):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    halftone = np.zeros_like(gray)
    for i in range(0, gray.shape[0], strength):
        for j in range(0, gray.shape[1], strength):
            if i + strength < gray.shape[0] and j + strength < gray.shape[1]:
                block = gray[i:i + strength, j:j + strength]
                avg = np.mean(block)
                radius = int((255 - avg) / 255 * strength / 2)
                cv2.circle(halftone, (j + strength // 2, i + strength // 2),
                           radius, (255, 255, 255), -1)
    halftone = cv2.cvtColor(halftone, cv2.COLOR_GRAY2BGR)
    return np.where(mask[:, :, np.newaxis], halftone, image)


def original_saturation(image, mask, strength):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    with np.errstate(over="ignore"):
        hsv[:, :, 1] = np.where(mask, hsv[:, :, 1] * strength, hsv[:, :, 1])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# --- matière d'essai --------------------------------------------------------

@pytest.fixture
def photo():
    """Une image à dégradé et taches, plus représentative qu'un bruit pur."""
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:256, 0:256]
    base = ((x + y) / 2).astype(np.uint8)
    image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR).astype(np.int16)
    image[:, :, 0] += 40
    image[:, :, 2] -= 30
    blobs = rng.integers(0, 255, (16, 16, 3))
    image += cv2.resize(blobs.astype(np.uint8), (256, 256),
                        interpolation=cv2.INTER_CUBIC).astype(np.int16) // 3
    return np.clip(image, 0, 255).astype(np.uint8)


@pytest.fixture
def everywhere():
    return np.ones((256, 256), dtype=bool)


class TestHalftone:
    """La trame vectorisée doit rendre la même chose que la double boucle."""

    @pytest.mark.parametrize("cell", [4, 6, 8, 12])
    def test_matches_the_original_where_it_drew(self, photo, everywhere, cell):
        mine = apply_halftone(photo, everywhere, cell)
        theirs = original_halftone(photo, everywhere, cell)

        # L'original sautait la dernière rangée et la dernière colonne de
        # cellules, les laissant non tramées ; on compare donc la zone qu'il
        # couvrait réellement.
        covered = (photo.shape[0] // cell - 1) * cell
        a = mine[:covered, :covered]
        b = theirs[:covered, :covered]

        differing = np.mean(np.any(a != b, axis=2))
        assert differing < 0.02, f"{differing:.1%} de pixels diffèrent"

    def test_no_dark_band_when_the_cell_divides_the_image(self):
        """Quand la cellule divise la taille, la trame va jusqu'au bord.

        C'est là que l'original perdait une rangée entière : sa boucle exigeait
        `i + cell < h` strictement, donc sautait la dernière même lorsqu'elle
        tombait juste.
        """
        cell, size = 8, 256  # 256 = 32 × 8, division exacte
        grey = np.full((size, size, 3), 128, dtype=np.uint8)
        everywhere = np.ones((size, size), dtype=bool)

        theirs = original_halftone(grey, everywhere, cell)
        mine = apply_halftone(grey, everywhere, cell)

        band = slice(size - cell, None)
        assert np.all(theirs[band, band] == 0), "l'original noircissait ce bord"
        assert np.any(mine[band, band] != 0), "le nouveau moteur doit y tramer"

    @pytest.mark.parametrize("size,cell", [(256, 12), (300, 7), (1024, 9)])
    def test_any_remaining_band_stays_under_one_cell(self, size, cell):
        """Une cellule partielle peut rester sombre, jamais davantage.

        Son point est centré dans la cellule complétée, dont le centre peut
        tomber hors de l'image. L'écart est borné à une cellule, contre une
        rangée entière systématiquement perdue dans l'original.
        """
        grey = np.full((size, size, 3), 128, dtype=np.uint8)
        mine = apply_halftone(grey, np.ones((size, size), bool), cell)

        drawn_rows = np.where(np.any(mine[:, :, 0] != 0, axis=1))[0]
        assert len(drawn_rows) > 0
        assert size - 1 - drawn_rows.max() < cell

    def test_is_fast_enough_for_print_size(self, everywhere):
        """À 6000 px la double boucle d'origine est hors de portée."""
        import time
        big = np.full((3000, 3000, 3), 128, dtype=np.uint8)
        mask = np.ones((3000, 3000), dtype=bool)
        start = time.perf_counter()
        apply_halftone(big, mask, 8)
        assert time.perf_counter() - start < 10.0


class TestSaturation:
    """Le débordement uint8 est du matériau : seize œuvres en dépendent."""

    @pytest.mark.parametrize("factor", [1.5, 2.0, 3.0])
    def test_overflow_matches_the_original(self, photo, everywhere, factor):
        mine = apply_saturate(photo, everywhere, factor, overflow=True)
        theirs = original_saturation(photo, everywhere, factor)
        assert np.array_equal(mine, theirs)

    def test_wraps_high_values_instead_of_clamping(self):
        """150 × 2 vaut 300, qui repasse à 44 au lieu de saturer à 255.

        La lecture se fait à une unité près : le retour en BGR puis en HSV
        n'est pas exactement réversible, dans l'original comme ici.
        """
        hsv = np.zeros((1, 1, 3), dtype=np.uint8)
        hsv[0, 0] = (90, 150, 200)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        one = np.ones((1, 1), bool)

        wrapped = cv2.cvtColor(
            apply_saturate(image, one, 2.0, overflow=True), cv2.COLOR_BGR2HSV
        )[0, 0, 1]
        clamped = cv2.cvtColor(
            apply_saturate(image, one, 2.0, overflow=False), cv2.COLOR_BGR2HSV
        )[0, 0, 1]

        assert abs(int(wrapped) - 44) <= 1, f"attendu ~44, obtenu {wrapped}"
        assert clamped == 255
        assert wrapped < clamped, "le débordement doit assombrir, pas saturer"

    def test_leaves_the_unmasked_area_alone(self):
        """Hors du masque, rien ne bouge — à l'aller-retour HSV près."""
        flat = np.full((256, 256, 3), 100, dtype=np.uint8)
        flat[:, :, 2] = 200
        mask = np.zeros(flat.shape[:2], dtype=bool)
        mask[:128] = True
        out = apply_saturate(flat, mask, 2.0)
        assert np.array_equal(out[128:], flat[128:])
        assert not np.array_equal(out[:128], flat[:128])
