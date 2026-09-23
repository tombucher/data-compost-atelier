"""Les routes que l'interface interroge pour se montrer.

La pellicule et les vignettes des tirages ne changent rien à une image :
elles servent à voir. Mais une pellicule qui ne couvrirait pas l'axe entier,
ou un aperçu de tirage qui servirait n'importe quel fichier du disque,
tromperait ou exposerait. Les images sont fabriquées ici : ces tests ne
dépendent d'aucune photographie.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from atelier.engine import Recipe
from atelier.metadata import save_with_recipe
from atelier.server import PELLICULE_IMAGES, Handler, Library


def fabriquer(chemin, graine):
    """Une image avec du relief, pour que la saillance ait prise."""
    rng = np.random.default_rng(graine)
    image = np.zeros((300, 400, 3), np.uint8)
    image[:] = rng.integers(40, 120, 3)
    for _ in range(6):
        centre = tuple(int(v) for v in rng.integers(40, 260, 2))
        cv2.circle(image, centre, int(rng.integers(15, 60)),
                   tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    cv2.imwrite(str(chemin), image)
    return str(chemin)


@pytest.fixture(scope="module")
def atelier(tmp_path_factory):
    images = tmp_path_factory.mktemp("images")
    sorties = tmp_path_factory.mktemp("sorties")
    sources = [fabriquer(images / f"essai_{i}.jpg", i) for i in range(3)]

    Handler.library = Library(images)
    Handler.output_dir = sorties.resolve()
    serveur = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    fil = threading.Thread(target=serveur.serve_forever, daemon=True)
    fil.start()
    yield f"http://127.0.0.1:{serveur.server_address[1]}", sources, sorties
    serveur.shutdown()
    serveur.server_close()


def poster(base, route, corps):
    requete = urllib.request.Request(
        base + route, data=json.dumps(corps).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(requete) as reponse:
        return json.loads(reponse.read())


@pytest.mark.parametrize("enchaine", [False, True])
def test_la_pellicule_couvre_l_axe_entier(atelier, enchaine):
    """Du premier instant au dernier : la pellicule montre où va l'axe."""
    base, sources, _ = atelier
    d = poster(base, "/api/pellicule",
               {"sources": sources, "seed": 3, "frames": 48, "chained": enchaine})

    assert d["rangs"][0] == 0
    assert d["rangs"][-1] == d["total"] - 1
    assert len(d["images"]) == len(d["rangs"]) == PELLICULE_IMAGES
    assert d["rangs"] == sorted(set(d["rangs"]))
    assert all(i.startswith("data:image/jpeg;base64,") for i in d["images"])


def test_un_axe_court_n_invente_pas_d_images(atelier):
    """Huit instants, huit vignettes au plus : pas de doublons pour remplir."""
    base, sources, _ = atelier
    d = poster(base, "/api/pellicule", {"sources": sources, "seed": 3, "frames": 8})
    assert len(d["rangs"]) == len(set(d["rangs"])) <= 8


def test_un_tirage_se_montre_par_son_nom(atelier):
    base, sources, sorties = atelier
    recette = Recipe(sources=tuple(sources), seed=9)
    save_with_recipe(np.full((900, 900, 3), 128, np.uint8),
                     sorties / "atelier_9_900.jpg", recette)

    with urllib.request.urlopen(base + "/api/tirage?name=atelier_9_900.jpg") as r:
        vignette = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
    assert vignette is not None and max(vignette.shape[:2]) <= 220


@pytest.mark.parametrize("nom", ["../images/essai_0.jpg", "inconnu.jpg", "/etc/hosts"])
def test_rien_d_autre_que_les_tirages_ne_sort(atelier, nom):
    """Le nom est ramené à son dernier segment, et doit être un tirage."""
    base, _, _ = atelier
    with pytest.raises(urllib.error.HTTPError) as refus:
        urllib.request.urlopen(base + "/api/tirage?name=" + urllib.request.quote(nom))
    assert refus.value.code == 404


def test_les_tirages_viennent_du_plus_recent(atelier):
    """Trier par nom mettait la graine 99 devant la graine 1234."""
    base, sources, sorties = atelier
    recette = Recipe(sources=tuple(sources), seed=1)
    ancien = save_with_recipe(np.zeros((50, 50, 3), np.uint8),
                              sorties / "atelier_99_50.jpg", recette)
    recent = save_with_recipe(np.zeros((50, 50, 3), np.uint8),
                              sorties / "atelier_1234_50.jpg", recette)
    maintenant = time.time()
    os.utime(ancien, (maintenant - 100, maintenant - 100))
    os.utime(recent, (maintenant, maintenant))

    with urllib.request.urlopen(base + "/api/exports") as r:
        noms = [t["name"] for t in json.loads(r.read())["exports"]]
    assert noms.index("atelier_1234_50.jpg") < noms.index("atelier_99_50.jpg")
