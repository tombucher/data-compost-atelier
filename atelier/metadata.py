"""La recette voyage dans l'image, pas dans un fichier à côté.

Le projet parle de se délester de fichiers : écrire un `.json` de paramètres
auprès de chaque image reviendrait à doubler leur nombre. La graine et les
réglages sont donc rangés dans l'image elle-même — bloc de texte pour un PNG,
champ EXIF pour un JPEG. Une image reste un fichier, et porte de quoi se
refaire en grand.

Une image sans recette se lit quand même : `read_recipe` renvoie None plutôt
que de lever. Un fichier venu d'ailleurs, ou passé par un logiciel qui a
effacé ses métadonnées, reste une image valable.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, PngImagePlugin

from atelier.engine import Effect, Recipe

# Clé sous laquelle la recette est rangée, dans un cas comme dans l'autre
MARKER = "data-compost-atelier"

# ImageDescription : champ EXIF de texte libre, bien plus largement conservé
# par les logiciels que UserComment, et lisible sans décodage particulier.
EXIF_IMAGE_DESCRIPTION = 0x010E


def recipe_to_json(recipe: Recipe, **extra) -> str:
    payload = {
        "version": 1,
        "seed": recipe.seed,
        "sources": [Path(s).name for s in recipe.sources],
        "effects": [{"name": e.name, "strength": e.strength} for e in recipe.effects],
        "per_image": [
            None if group is None
            else [{"name": e.name, "strength": e.strength} for e in group]
            for group in recipe.per_image
        ],
        "saliency_threshold": recipe.saliency_threshold,
        "smoothness": recipe.smoothness,
        "edge_blur": recipe.edge_blur,
        "saturation_overflow": recipe.saturation_overflow,
        **extra,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def recipe_from_json(raw: str, sources=None) -> Recipe:
    """Reconstruit une recette.

    Seuls les noms de fichiers sont enregistrés, pas les chemins : une image
    reste lisible quand le dossier a bougé. `sources` permet de fournir les
    chemins retrouvés au moment de refaire le rendu.
    """
    data = json.loads(raw)
    return Recipe(
        sources=tuple(sources if sources is not None else data.get("sources", ())),
        seed=int(data["seed"]),
        effects=tuple(
            Effect(e["name"], float(e["strength"])) for e in data.get("effects", ())
        ),
        per_image=tuple(
            None if group is None
            else tuple(Effect(e["name"], float(e["strength"])) for e in group)
            for group in data.get("per_image", ())
        ),
        saliency_threshold=int(data.get("saliency_threshold", 60)),
        smoothness=float(data.get("smoothness", 3.0)),
        edge_blur=int(data.get("edge_blur", 0)),
        saturation_overflow=bool(data.get("saturation_overflow", True)),
    )


def save_with_recipe(
    image: np.ndarray, path: str | Path, recipe: Recipe, quality: int = 95, **extra
) -> Path:
    """Écrit l'image avec sa recette à l'intérieur.

    `image` est en BGR, la convention d'OpenCV utilisée par le moteur.
    """
    path = Path(path)
    payload = recipe_to_json(recipe, **extra)
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if path.suffix.lower() == ".png":
        info = PngImagePlugin.PngInfo()
        info.add_text(MARKER, payload)
        pil.save(path, "PNG", pnginfo=info)
    else:
        exif = pil.getexif()
        exif[EXIF_IMAGE_DESCRIPTION] = f"{MARKER} {payload}"
        pil.save(path, "JPEG", quality=quality, exif=exif)

    return path


def read_recipe(path: str | Path, sources=None) -> Recipe | None:
    """Relit la recette d'une image, ou None si elle n'en porte pas."""
    path = Path(path)
    try:
        with Image.open(path) as pil:
            raw = pil.info.get(MARKER)
            if raw is None:
                described = pil.getexif().get(EXIF_IMAGE_DESCRIPTION)
                if isinstance(described, str) and described.startswith(MARKER):
                    raw = described[len(MARKER):].strip()
            if not raw:
                return None
            return recipe_from_json(raw, sources=sources)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        # Image absente, illisible, ou métadonnées abîmées : ce n'est pas une
        # erreur, seulement une image dont on ne sait pas refaire le tirage.
        return None
