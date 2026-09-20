"""Serveur local de l'atelier.

L'outil est une application web, mais rien n'est hébergé : un petit serveur
tourne sur la machine, le navigateur porte l'interface, et les images ne
quittent jamais le disque. Les effets restent en Python parce qu'ils reposent
sur OpenCV et qu'ils sont déjà écrits et testés.

Le partage des rôles : le navigateur règle et affiche, le serveur calcule.
Un aperçu se rend en basse définition pour rester vif pendant qu'on bouge un
curseur ; le tirage en grand format part sur demande.

    python -m atelier.server ~/Images/pavots

La bibliothèque http.server de la bibliothèque standard suffit : un seul
utilisateur, en local. Pas de dépendance de plus à installer sur une machine
qui sert déjà à faire des images.
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from atelier.chance import random_recipe
from atelier.decay import Decay, compose_at, count_frames, sequence
from atelier.engine import (
    EFFECTS,
    HALFTONE_SHAPES,
    Effect,
    Recipe,
    load_source,
    new_seed,
    rank_field,
    render,
    spot_field,
)
from atelier.metadata import read_payload, read_recipe, save_with_recipe
from atelier.video import write_video

logger = logging.getLogger("atelier")

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Taille de l'aperçu. Assez grande pour juger, assez petite pour suivre un
# curseur : environ deux dixièmes de seconde sur quatre images.
PREVIEW = 900

# Taille des vignettes qui montrent ce que fait chaque matière. Une matière
# qu'on active à l'aveugle est une matière qu'on n'essaie pas ; à cette
# taille, les sept se calculent en une centaine de millisecondes.
VIGNETTE = 132

# Forces employées pour ces vignettes : assez marquées pour se reconnaître au
# premier coup d'œil, ce qui n'est pas la même chose qu'un réglage utile.
VIGNETTE_FORCES = {
    "pixelate": 0.020, "bitmap": 0.95, "saturate": 3.0, "halftone": 0.030,
    "pixelsort": 0.09, "mosh": 0.07, "shift": 0.020,
}
THUMBNAIL = 220


class Library:
    """Les images du dossier de travail, chargées une fois pour toutes.

    Relire les fichiers à chaque mouvement de curseur coûterait bien plus cher
    que le rendu lui-même.
    """

    # Taille de la copie de travail. Réduire une photo de 5000 px vers 130
    # d'un seul coup coûte 25 fois plus cher que de passer par une étape
    # intermédiaire — l'aperçu payait ce prix à chaque mouvement de curseur.
    # Les tirages, eux, partent toujours de la pleine résolution.
    WORK_SIZE = 2048

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self._cache: dict[Path, np.ndarray] = {}
        self._work: dict[Path, np.ndarray] = {}
        self._lock = threading.Lock()

    def move_to(self, raw: str):
        """Change de dossier de travail sans relancer l'atelier.

        Le cache d'images est vidé : une série chargée peut peser lourd, et
        on ne revient pas forcément à la précédente.
        """
        target = Path(raw).expanduser().resolve()
        if not target.is_dir():
            raise ValueError(f"ce dossier n'existe pas : {raw}")
        with self._lock:
            self.root = target
            self._cache.clear()
            self._work.clear()

    def neighbours(self) -> list[dict]:
        """Dossiers voisins contenant des images, pour changer de série.

        Le parent et les sous-dossiers du dossier courant : de quoi passer
        d'une série à l'autre sans taper de chemin.
        """
        seen, found = set(), []
        candidates = [self.root.parent]
        try:
            candidates += sorted(p for p in self.root.iterdir() if p.is_dir())
            candidates += sorted(
                p for p in self.root.parent.iterdir()
                if p.is_dir() and p != self.root
            )
        except OSError:
            pass

        for path in candidates:
            if path in seen or not path.is_dir():
                continue
            seen.add(path)
            try:
                count = sum(
                    1 for f in path.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
                    and not f.name.startswith(".")
                )
            except OSError:
                continue
            if count:
                found.append({"path": str(path), "name": path.name or str(path),
                              "count": count})
        return found[:30]

    def listing(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        found = [
            p for p in sorted(self.root.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            and not p.name.startswith(".")
        ]
        return [{"name": p.name, "path": str(p)} for p in found]

    def resolve(self, raw: str) -> Path:
        """Refuse tout chemin hors du dossier de travail."""
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"fichier absent : {raw}")
        if self.root not in path.parents and path.parent != self.root:
            raise ValueError("chemin hors du dossier de travail")
        return path

    def image(self, raw: str, for_size: int | None = None) -> np.ndarray:
        """L'image source, ou une copie de travail quand la cible est petite.

        Un rendu jusqu'à la moitié de `WORK_SIZE` ne perd rien de visible à
        partir de la copie : il allait de toute façon réduire davantage. Au
        delà, la pleine résolution est servie, pour que les tirages gardent
        tout leur détail.
        """
        path = self.resolve(raw)
        with self._lock:
            if path not in self._cache:
                self._cache[path] = load_source(path)
            pleine = self._cache[path]

            if for_size is None or for_size * 2 > self.WORK_SIZE:
                return pleine
            if max(pleine.shape[:2]) <= self.WORK_SIZE:
                return pleine
            if path not in self._work:
                facteur = self.WORK_SIZE / max(pleine.shape[:2])
                self._work[path] = cv2.resize(
                    pleine,
                    (max(1, int(pleine.shape[1] * facteur)),
                     max(1, int(pleine.shape[0] * facteur))),
                    interpolation=cv2.INTER_AREA,
                )
            return self._work[path]


def decay_from_request(body: dict) -> Decay:
    """L'axe du temps et la manière dont la matière se défait."""
    defaut = Decay()

    def nombre(cle, mini, maxi, conversion=float):
        brut = body.get(cle)
        if brut is None:
            return getattr(defaut, cle)
        return max(mini, min(maxi, conversion(brut)))

    def bascule(cle):
        brut = body.get(cle)
        return getattr(defaut, cle) if brut is None else bool(brut)

    return Decay(
        frames=nombre("frames", 2, 600, int),
        fps=nombre("fps", 1, 60, int),
        stagger=nombre("stagger", 0.0, 1.0),
        residue=nombre("residue", 0.0, 0.9),
        ripening=nombre("ripening", 0.0, 6.0),
        tearing=nombre("tearing", 0.0, 1.0),
        min_segment=nombre("min_segment", 0.002, 0.5),
        max_segment=nombre("max_segment", 0.004, 0.8),
        segment_growth=nombre("segment_growth", 0.0, 6.0),
        channel_shift=nombre("channel_shift", 0.0, 0.2),
        enhanced=bascule("enhanced"),
        channel_quirk=bascule("channel_quirk"),
        sort_visible=bascule("sort_visible"),
        even_dissolve=bascule("even_dissolve"),
    )


def decay_payload(decay: Decay) -> dict:
    """L'axe tel que l'interface l'attend."""
    return {
        "frames": decay.frames, "fps": decay.fps,
        "stagger": decay.stagger, "residue": decay.residue,
        "ripening": decay.ripening, "tearing": decay.tearing,
        "min_segment": decay.min_segment, "max_segment": decay.max_segment,
        "segment_growth": decay.segment_growth,
        "channel_shift": decay.channel_shift,
        "enhanced": decay.enhanced, "channel_quirk": decay.channel_quirk,
        "sort_visible": decay.sort_visible, "even_dissolve": decay.even_dissolve,
    }


# Les tirages vidéo durent des minutes : ils partent en tâche de fond et
# l'interface suit leur avancement, plutôt que de laisser le navigateur en
# attente sur une requête qui n'aboutit pas.
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def run_video_job(job_id: str, recipe, decay, size, images, target: Path):
    total = count_frames(decay)

    def frames():
        for done, frame in enumerate(sequence(recipe, decay, size, images), 1):
            with JOBS_LOCK:
                JOBS[job_id]["done"] = done
            yield frame

    try:
        written = write_video(frames(), target, decay.fps)
        with JOBS_LOCK:
            JOBS[job_id].update(state="fini", path=str(target), written=written)
        logger.info("vidéo écrite : %s (%d frames)", target, written)
    except Exception as exc:  # noqa: BLE001 — l'échec appartient au travail
        logger.warning("vidéo échouée : %s", exc)
        with JOBS_LOCK:
            JOBS[job_id].update(state="échec", error=str(exc))


def recipe_from_request(body: dict, library: Library,
                        for_size: int | None = None) -> tuple[Recipe, list]:
    """Construit une recette et charge ses images.

    `for_size` dit à quelle taille on va rendre : une copie de travail suffit
    pour un aperçu, un tirage veut la pleine résolution.
    """
    paths = [str(library.resolve(p)) for p in body.get("sources", [])]
    if not paths:
        raise ValueError("aucune image choisie")

    def matiere(e) -> Effect:
        return Effect(e["name"], float(e["strength"]), e.get("shape", "round"),
                      float(e.get("angle", 0.0)), bool(e.get("cross", False)))

    effects = tuple(matiere(e) for e in body.get("effects", []) if e.get("name"))

    def group(raw):
        """Un réglage propre à une image, ou None pour suivre le commun."""
        if raw is None:
            return None
        return tuple(matiere(e) for e in raw if e.get("name"))

    recipe = Recipe(
        sources=tuple(paths),
        seed=int(body.get("seed") or new_seed()),
        effects=effects,
        per_image=tuple(group(g) for g in body.get("per_image", [])),
        saliency_threshold=int(body.get("threshold", 60)),
        bleed=float(body.get("bleed", 0.0)),
        full_frame=bool(body.get("full_frame", False)),
        smoothness=float(body.get("smoothness", 3.0)),
        edge_blur=int(body.get("edge_blur", 0)),
        saturation_overflow=bool(body.get("overflow", True)),
    )
    return recipe, [library.image(p, for_size=for_size) for p in paths]


AXE_KEYS = tuple(decay_payload(Decay())) + ("moment",)


def _matiere_json(effect: Effect) -> dict:
    return {"name": effect.name, "strength": effect.strength,
            "shape": effect.shape, "angle": effect.angle,
            "cross": effect.cross}


def recipe_payload(recipe, sources=None, extra: dict | None = None) -> dict:
    """La recette telle que l'interface l'attend, axe compris."""
    axe = {k: extra[k] for k in AXE_KEYS if extra and k in extra}
    return {
        **axe,
        "sources": list(recipe.sources if sources is None else sources),
        "seed": recipe.seed,
        "effects": [_matiere_json(e) for e in recipe.effects],
        "per_image": [
            None if g is None else [_matiere_json(e) for e in g]
            for g in recipe.per_image
        ],
        "threshold": recipe.saliency_threshold,
        "bleed": recipe.bleed,
        "full_frame": recipe.full_frame,
        "smoothness": recipe.smoothness,
        "edge_blur": recipe.edge_blur,
        "overflow": recipe.saturation_overflow,
    }


class Handler(BaseHTTPRequestHandler):
    library: Library
    output_dir: Path

    # --- plomberie ---------------------------------------------------------

    def log_message(self, fmt, *args):  # noqa: A003 — signature imposée
        logger.debug(fmt, *args)

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, code: int = 200):
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _fail(self, exc: Exception, code: int = 400):
        logger.warning("%s", exc)
        self._json({"error": str(exc)}, code)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    @staticmethod
    def _jpeg(image: np.ndarray, quality: int = 88) -> bytes:
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("encodage JPEG impossible")
        return buffer.tobytes()

    # --- routes ------------------------------------------------------------

    def do_GET(self):  # noqa: N802 — signature imposée
        route = urlparse(self.path)
        query = parse_qs(route.query)
        try:
            if route.path in ("/", "/index.html"):
                return self._static("index.html")
            if route.path == "/api/sources":
                return self._json({
                    "root": str(self.library.root),
                    "images": self.library.listing(),
                    "neighbours": self.library.neighbours(),
                    "output": str(self.output_dir.resolve()),
                })
            if route.path == "/api/thumbnail":
                return self._thumbnail(unquote(query.get("path", [""])[0]))
            if route.path == "/api/recipe":
                return self._recipe_of(unquote(query.get("path", [""])[0]))
            if route.path == "/api/exports":
                return self._exports()
            if route.path == "/api/job":
                return self._job(query.get("id", [""])[0])
            if route.path == "/api/chance":
                return self._chance(query.get("family", [""])[0])
            if route.path == "/api/motif":
                return self._motif(query.get("shape", ["round"])[0])
            return self._static(route.path.lstrip("/"))
        except Exception as exc:  # noqa: BLE001 — une requête ne tue pas le serveur
            self._fail(exc)

    def do_POST(self):  # noqa: N802
        route = urlparse(self.path)
        try:
            if route.path == "/api/render":
                return self._render()
            if route.path == "/api/vignettes":
                return self._vignettes()
            if route.path == "/api/export":
                return self._export()
            if route.path == "/api/folder":
                return self._folder()
            if route.path == "/api/video":
                return self._video()
            self._json({"error": "route inconnue"}, 404)
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)

    # --- mise en œuvre -----------------------------------------------------

    def _static(self, name: str):
        target = (WEB_ROOT / name).resolve()
        if WEB_ROOT not in target.parents or not target.is_file():
            return self._json({"error": "introuvable"}, 404)
        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), kind)

    def _thumbnail(self, raw: str):
        image = self.library.image(raw)
        height, width = image.shape[:2]
        scale = THUMBNAIL / max(height, width)
        small = cv2.resize(
            image, (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        self._send(200, self._jpeg(small, 75), "image/jpeg")

    def _recipe_of(self, raw: str):
        """Relit la recette cachée dans une image déjà produite."""
        path = Path(raw).expanduser()
        found = read_recipe(path)
        if found is None:
            return self._json({"recipe": None})
        self._json({"recipe": recipe_payload(found, extra=read_payload(path))})

    def _exports(self):
        """Les tirages déjà faits, avec la recette que chacun porte.

        C'est ici que la graine sert enfin à quelque chose : on reprend un
        tirage, on retrouve ses réglages, on le relance en plus grand.
        """
        if not self.output_dir.is_dir():
            return self._json({"exports": []})

        found = []
        for path in sorted(self.output_dir.iterdir(), reverse=True):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            recipe = read_recipe(path)
            if recipe is None:
                continue
            # Les noms enregistrés sont retrouvés dans le dossier de travail :
            # une recette reste jouable même si le tirage a été déplacé.
            sources, complete = [], True
            for name in recipe.sources:
                candidate = self.library.root / name
                if candidate.is_file():
                    sources.append(str(candidate))
                else:
                    complete = False
            found.append({
                "name": path.name,
                "seed": recipe.seed,
                "complete": complete,
                "recipe": recipe_payload(recipe, sources, read_payload(path)),
            })
        self._json({"exports": found[:40]})

    def _chance(self, family: str):
        """Tire une recette dans les habitudes du corpus conservé.

        Le tirage ne rend rien : il renvoie des réglages, que l'interface
        applique comme si on les avait posés à la main. Ce qui suit reste
        modifiable, et la graine permet d'y revenir.
        """
        paths = [entry["path"] for entry in self.library.listing()]
        if not paths:
            return self._json({"error": "aucune image dans ce dossier"}, 400)

        recipe = random_recipe(paths, family=family or None)
        payload = recipe_payload(recipe)
        payload["family"] = family or "au hasard"
        self._json({"recipe": payload})

    def _motif(self, shape: str):
        """Le motif d'une forme de trame, à mi-densité.

        Dessiné à partir du champ que le moteur emploie réellement, pas d'une
        icône redessinée à la main : ce qu'on voit dans le menu est ce que la
        trame produira.
        """
        shape = shape if shape in HALFTONE_SHAPES else "round"
        cellule, tuiles = 16, 3
        champ = spot_field(cellule, shape) if shape == "round" \
            else rank_field(cellule, shape)
        seuil = cellule / 4 if shape == "round" else 0.5
        motif = np.tile((champ <= seuil).astype(np.uint8) * 255, (tuiles, tuiles))
        cote = cellule * tuiles
        image = cv2.resize(motif, (cote * 2, cote * 2),
                           interpolation=cv2.INTER_NEAREST)
        ok, encode = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("motif illisible")
        self._send(200, encode.tobytes(), "image/png")

    def _vignettes(self):
        """Ce que fait chaque matière, sur les images en cours.

        Chacune est rendue seule, à force marquée, pour qu'on reconnaisse son
        geste. Ce ne sont pas des aperçus du réglage courant mais des
        échantillons de matière — comme on montre un nuancier, pas un mur
        peint.
        """
        body = self._body()
        recipe, images = recipe_from_request(body, self.library,
                                             for_size=VIGNETTE)

        # Les huit vignettes partagent les mêmes cartes de saillance : les
        # recalculer huit fois coûterait plus cher que les huit rendus.
        from atelier.decay import prepare_fields
        cartes = prepare_fields(recipe, images)

        rendus = {"nu": self._jpeg64(
            render(recipe, size=VIGNETTE, sources=images, fields=cartes))}
        for nom in EFFECTS:
            essai = Recipe(
                sources=recipe.sources, seed=recipe.seed,
                effects=(Effect(nom, VIGNETTE_FORCES[nom]),),
                saliency_threshold=recipe.saliency_threshold,
                smoothness=recipe.smoothness, edge_blur=recipe.edge_blur,
                # Les tris ne se voient pas s'ils restent dans la zone calme
                bleed=0.85 if nom in ("pixelsort", "mosh") else recipe.bleed,
                full_frame=recipe.full_frame,
            )
            rendus[nom] = self._jpeg64(
                render(essai, size=VIGNETTE, sources=images, fields=cartes))
        self._json({"vignettes": rendus})

    def _jpeg64(self, image) -> str:
        import base64
        return "data:image/jpeg;base64," + base64.b64encode(
            self._jpeg(image, 78)).decode("ascii")

    def _render(self):
        """L'aperçu, à l'instant demandé de l'axe.

        Il n'y a plus de rendu « fixe » distinct d'une frame : l'instant 0 est
        la composition intacte, et tout le reste est le même calcul plus loin
        dans le temps.
        """
        body = self._body()
        size = int(body.get("size") or PREVIEW)
        recipe, images = recipe_from_request(body, self.library, for_size=size)
        decay = decay_from_request(body)
        index = max(0, min(int(body.get("moment", 0)), count_frames(decay) - 1))

        image = compose_at(recipe, decay, size, index, images)

        payload = self._jpeg(image)
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(payload)))
        # La graine voyage dans l'en-tête : l'interface l'affiche sans
        # avoir à la deviner quand elle a laissé le serveur en tirer une.
        self.send_header("X-Atelier-Seed", str(recipe.seed))
        self.send_header("X-Atelier-Frames", str(count_frames(decay)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _folder(self):
        """Change de dossier de travail."""
        self.library.move_to(self._body().get("path", ""))
        logger.info("dossier : %s", self.library.root)
        self._json({
            "root": str(self.library.root),
            "images": self.library.listing(),
            "neighbours": self.library.neighbours(),
            "output": str(self.output_dir.resolve()),
        })

    def _video(self):
        import uuid

        body = self._body()
        recipe, images = recipe_from_request(body, self.library)
        decay = decay_from_request(body)
        size = int(body.get("size", 2000))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / f"atelier_{recipe.seed}_{size}.mp4"
        job_id = uuid.uuid4().hex[:12]
        with JOBS_LOCK:
            JOBS[job_id] = {"state": "en cours", "done": 0,
                            "total": count_frames(decay)}

        threading.Thread(
            target=run_video_job,
            args=(job_id, recipe, decay, size, images, target),
            daemon=True,
        ).start()
        self._json({"job": job_id, "total": count_frames(decay),
                    "seed": recipe.seed})

    def _job(self, job_id: str):
        with JOBS_LOCK:
            state = JOBS.get(job_id)
        self._json(state or {"state": "inconnu"})

    def _export(self):
        body = self._body()
        recipe, images = recipe_from_request(body, self.library)
        decay = decay_from_request(body)
        size = int(body.get("size", 4000))
        index = max(0, min(int(body.get("moment", 0)), count_frames(decay) - 1))
        image = compose_at(recipe, decay, size, index, images)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".png" if body.get("format") == "png" else ".jpg"
        # L'instant figure dans le nom : deux tirages du même axe se
        # distinguent sans avoir à ouvrir les fichiers.
        marque = "" if index == 0 else f"_t{index}"
        target = self.output_dir / f"atelier_{recipe.seed}{marque}_{size}{suffix}"
        save_with_recipe(image, target, recipe, rendered_at=size, moment=index,
                         **decay_payload(decay))

        logger.info("tirage écrit : %s", target)
        self._json({
            "path": str(target),
            "seed": recipe.seed,
            "size": size,
            "moment": index,
            "centimetres": round(size / 300 * 2.54, 1),
        })


DEFAULT_PORT = 8771


def open_server(port: int, attempts: int = 10):
    """Ouvre le serveur sur le premier port libre à partir de `port`.

    Un port déjà pris — le plus souvent un atelier resté ouvert dans un autre
    terminal — ne doit pas se solder par une trace d'erreur Python.
    """
    last = None
    for candidate in range(port, port + attempts):
        try:
            return ThreadingHTTPServer(("127.0.0.1", candidate), Handler), candidate
        except OSError as exc:
            last = exc
            if candidate == port:
                logger.info("port %d déjà pris, je prends le suivant", port)
    raise SystemExit(
        f"Aucun port libre entre {port} et {port + attempts - 1} ({last}).\n"
        f"Fermez l'atelier déjà ouvert, ou choisissez un port : --port 9000"
    )


def serve(root: Path, output: Path, port: int = DEFAULT_PORT, open_browser: bool = True):
    Handler.library = Library(root)
    Handler.output_dir = output.expanduser().resolve()

    server, port = open_server(port)
    address = f"http://127.0.0.1:{port}/"

    logger.info("atelier sur %s", address)
    logger.info("images   : %s", Handler.library.root)
    logger.info("tirages  : %s", Handler.output_dir)

    if not Handler.library.root.is_dir():
        logger.warning("ce dossier d'images n'existe pas")
    elif not Handler.library.listing():
        logger.warning("aucune image lisible dans ce dossier")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("arrêt")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Atelier de composition")
    parser.add_argument("images", nargs="?", default=".",
                        help="dossier des images sources")
    parser.add_argument("--sorties", default="sorties",
                        help="dossier où écrire les tirages")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--sans-navigateur", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    serve(Path(args.images), Path(args.sorties), args.port, not args.sans_navigateur)


if __name__ == "__main__":
    main()
