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

from atelier.engine import (
    Effect,
    PREVIEW_SIZE,
    Recipe,
    load_source,
    new_seed,
    render,
)
from atelier.metadata import read_recipe, save_with_recipe
from atelier.video import (
    Motion,
    count_frames,
    render_frame,
    render_sequence,
    write_video,
)

logger = logging.getLogger("atelier")

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Taille de l'aperçu. Assez grande pour juger, assez petite pour suivre un
# curseur : environ deux dixièmes de seconde sur quatre images.
PREVIEW = 900
THUMBNAIL = 220


class Library:
    """Les images du dossier de travail, chargées une fois pour toutes.

    Relire les fichiers à chaque mouvement de curseur coûterait bien plus cher
    que le rendu lui-même.
    """

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self._cache: dict[Path, np.ndarray] = {}
        self._lock = threading.Lock()

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

    def image(self, raw: str) -> np.ndarray:
        path = self.resolve(raw)
        with self._lock:
            if path not in self._cache:
                self._cache[path] = load_source(path)
            return self._cache[path]


def motion_from_request(body: dict) -> Motion:
    return Motion(
        frames_per_transition=max(2, int(body.get("frames", 24))),
        fps=max(1, int(body.get("fps", 24))),
    )


# Les tirages vidéo durent des minutes : ils partent en tâche de fond et
# l'interface suit leur avancement, plutôt que de laisser le navigateur en
# attente sur une requête qui n'aboutit pas.
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def run_video_job(job_id: str, recipe, motion, size, images, target: Path):
    total = count_frames(recipe, motion)

    def frames():
        for done, frame in enumerate(render_sequence(recipe, motion, size, images), 1):
            with JOBS_LOCK:
                JOBS[job_id]["done"] = done
            yield frame

    try:
        written = write_video(frames(), target, motion.fps)
        with JOBS_LOCK:
            JOBS[job_id].update(state="fini", path=str(target), written=written)
        logger.info("vidéo écrite : %s (%d frames)", target, written)
    except Exception as exc:  # noqa: BLE001 — l'échec appartient au travail
        logger.warning("vidéo échouée : %s", exc)
        with JOBS_LOCK:
            JOBS[job_id].update(state="échec", error=str(exc))


def recipe_from_request(body: dict, library: Library) -> tuple[Recipe, list]:
    """Construit une recette et charge ses images."""
    paths = [str(library.resolve(p)) for p in body.get("sources", [])]
    if not paths:
        raise ValueError("aucune image choisie")

    effects = tuple(
        Effect(e["name"], float(e["strength"]))
        for e in body.get("effects", [])
        if e.get("name")
    )
    recipe = Recipe(
        sources=tuple(paths),
        seed=int(body.get("seed") or new_seed()),
        effects=effects,
        saliency_threshold=int(body.get("threshold", 120)),
        smoothness=float(body.get("smoothness", 3.0)),
        edge_blur=int(body.get("edge_blur", 0)),
        saturation_overflow=bool(body.get("overflow", True)),
    )
    return recipe, [library.image(p) for p in paths]


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
                })
            if route.path == "/api/thumbnail":
                return self._thumbnail(unquote(query.get("path", [""])[0]))
            if route.path == "/api/recipe":
                return self._recipe_of(unquote(query.get("path", [""])[0]))
            if route.path == "/api/exports":
                return self._exports()
            if route.path == "/api/job":
                return self._job(query.get("id", [""])[0])
            return self._static(route.path.lstrip("/"))
        except Exception as exc:  # noqa: BLE001 — une requête ne tue pas le serveur
            self._fail(exc)

    def do_POST(self):  # noqa: N802
        route = urlparse(self.path)
        try:
            if route.path == "/api/render":
                return self._render()
            if route.path == "/api/export":
                return self._export()
            if route.path == "/api/frame":
                return self._frame()
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
        found = read_recipe(Path(raw).expanduser())
        if found is None:
            return self._json({"recipe": None})
        self._json({"recipe": {
            "seed": found.seed,
            "sources": list(found.sources),
            "effects": [{"name": e.name, "strength": e.strength} for e in found.effects],
            "threshold": found.saliency_threshold,
            "smoothness": found.smoothness,
            "edge_blur": found.edge_blur,
            "overflow": found.saturation_overflow,
        }})

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
                "recipe": {
                    "sources": sources,
                    "seed": recipe.seed,
                    "effects": [{"name": e.name, "strength": e.strength}
                                for e in recipe.effects],
                    "threshold": recipe.saliency_threshold,
                    "smoothness": recipe.smoothness,
                    "edge_blur": recipe.edge_blur,
                    "overflow": recipe.saturation_overflow,
                },
            })
        self._json({"exports": found[:40]})

    def _render(self):
        body = self._body()
        recipe, images = recipe_from_request(body, self.library)
        size = int(body.get("size") or PREVIEW)
        image = render(recipe, size=size, sources=images)

        payload = self._jpeg(image)
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(payload)))
        # La graine voyage dans l'en-tête : l'interface l'affiche sans
        # avoir à la deviner quand elle a laissé le serveur en tirer une.
        self.send_header("X-Atelier-Seed", str(recipe.seed))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _frame(self):
        """Une frame de la séquence, à la taille voulue.

        C'est ce qui remplace la capture d'écran : on parcourt, on s'arrête,
        on sort l'instant en pleine définition.
        """
        body = self._body()
        recipe, images = recipe_from_request(body, self.library)
        motion = motion_from_request(body)
        size = int(body.get("size") or PREVIEW)
        index = int(body.get("frame", 0))

        image = render_frame(recipe, motion, size, index, images)

        if body.get("save"):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            target = self.output_dir / f"atelier_{recipe.seed}_f{index}_{size}.jpg"
            save_with_recipe(image, target, recipe, rendered_at=size, frame=index)
            return self._json({
                "path": str(target), "seed": recipe.seed, "frame": index,
                "size": size, "centimetres": round(size / 300 * 2.54, 1),
            })

        payload = self._jpeg(image)
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Atelier-Seed", str(recipe.seed))
        self.send_header("X-Atelier-Frames", str(count_frames(recipe, motion)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _video(self):
        import uuid

        body = self._body()
        recipe, images = recipe_from_request(body, self.library)
        if len(images) < 2:
            raise ValueError("il faut au moins deux images pour une vidéo")
        motion = motion_from_request(body)
        size = int(body.get("size", 2000))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / f"atelier_{recipe.seed}_{size}.mp4"
        job_id = uuid.uuid4().hex[:12]
        with JOBS_LOCK:
            JOBS[job_id] = {"state": "en cours", "done": 0,
                            "total": count_frames(recipe, motion)}

        threading.Thread(
            target=run_video_job,
            args=(job_id, recipe, motion, size, images, target),
            daemon=True,
        ).start()
        self._json({"job": job_id, "total": count_frames(recipe, motion),
                    "seed": recipe.seed})

    def _job(self, job_id: str):
        with JOBS_LOCK:
            state = JOBS.get(job_id)
        self._json(state or {"state": "inconnu"})

    def _export(self):
        body = self._body()
        recipe, images = recipe_from_request(body, self.library)
        size = int(body.get("size", 4000))
        image = render(recipe, size=size, sources=images)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".png" if body.get("format") == "png" else ".jpg"
        target = self.output_dir / f"atelier_{recipe.seed}_{size}{suffix}"
        save_with_recipe(image, target, recipe, rendered_at=size)

        logger.info("tirage écrit : %s", target)
        self._json({
            "path": str(target),
            "seed": recipe.seed,
            "size": size,
            "centimetres": round(size / 300 * 2.54, 1),
        })


def serve(root: Path, output: Path, port: int = 8765, open_browser: bool = True):
    Handler.library = Library(root)
    Handler.output_dir = output.expanduser().resolve()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    address = f"http://127.0.0.1:{port}/"

    logger.info("atelier sur %s", address)
    logger.info("images   : %s", Handler.library.root)
    logger.info("tirages  : %s", Handler.output_dir)

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
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--sans-navigateur", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    serve(Path(args.images), Path(args.sorties), args.port, not args.sans_navigateur)


if __name__ == "__main__":
    main()
