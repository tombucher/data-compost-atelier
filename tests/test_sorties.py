"""Un tirage n'en écrase jamais un autre.

Le nom d'un tirage porte sa graine, son instant et sa taille. Ce n'est pas
de quoi l'identifier : la même graine rendue avec d'autres matières, un
autre seuil ou dans l'autre mode du temps donne une tout autre image. Deux
tirages portaient donc le même nom, et le second effaçait le premier au
moment d'écrire — sans rien demander, et sur du travail gardé.
"""
import threading

from atelier.server import NOMS_PROMIS, nom_libre


def test_le_premier_garde_le_nom_nu(tmp_path):
    assert nom_libre(tmp_path, "atelier_42_4000", ".jpg").name \
        == "atelier_42_4000.jpg"


def test_un_nom_deja_sur_le_disque_n_est_pas_repris(tmp_path):
    (tmp_path / "atelier_42_4000.jpg").write_bytes("un tirage gardé".encode())
    suivant = nom_libre(tmp_path, "atelier_42_4000", ".jpg")

    assert suivant.name == "atelier_42_4000_2.jpg"
    assert (tmp_path / "atelier_42_4000.jpg").read_bytes() == "un tirage gardé".encode()


def test_les_rangs_s_enchainent(tmp_path):
    for attendu in ("atelier_7_2000.jpg", "atelier_7_2000_2.jpg",
                    "atelier_7_2000_3.jpg"):
        chemin = nom_libre(tmp_path, "atelier_7_2000", ".jpg")
        assert chemin.name == attendu
        chemin.write_bytes(b"x")


def test_un_nom_promis_compte_comme_pris(tmp_path):
    """Le fichier n'existe qu'une fois écrit, et un tirage dure des minutes.

    Sans cela, deux tirages lancés coup sur coup choisiraient le même nom et
    le second écraserait le premier en arrivant.
    """
    premier = nom_libre(tmp_path, "atelier_9_4000", ".jpg")
    second = nom_libre(tmp_path, "atelier_9_4000", ".jpg")

    assert not premier.exists()
    assert premier != second
    assert second.name == "atelier_9_4000_2.jpg"


def test_deux_formats_ne_se_marchent_pas_dessus(tmp_path):
    (tmp_path / "atelier_5_4000.jpg").write_bytes(b"x")
    assert nom_libre(tmp_path, "atelier_5_4000", ".png").name \
        == "atelier_5_4000.png"


def test_deux_demandes_simultanees_donnent_deux_noms(tmp_path):
    """Deux navigateurs, ou deux clics pressés : chacun son fichier."""
    pris = []
    barriere = threading.Barrier(8)

    def demander():
        barriere.wait()
        pris.append(nom_libre(tmp_path, "atelier_1_1000", ".jpg"))

    fils = [threading.Thread(target=demander) for _ in range(8)]
    for f in fils:
        f.start()
    for f in fils:
        f.join()

    assert len(set(pris)) == 8


def test_l_instant_reste_dans_le_nom(tmp_path):
    """Deux instants du même axe se distinguent sans ouvrir les fichiers."""
    debut = nom_libre(tmp_path, "atelier_3_4000", ".jpg")
    plus_tard = nom_libre(tmp_path, "atelier_3_t24_4000", ".jpg")
    assert debut.name == "atelier_3_4000.jpg"
    assert plus_tard.name == "atelier_3_t24_4000.jpg"


def teardown_module(module):
    """Les promesses sont propres au serveur, pas aux tests."""
    NOMS_PROMIS.clear()
