# Sphinx plugin that stages every notebook under examples/ into the docs tree, extracts a thumbnail
# from its tagged or last image output, and emits the grid-card gallery page. Categories come from the
# subdirectory a notebook sits in; see docs/source/dev/docs.rst.
# Adapted from gEconpy, which adapted it from PyMC / seaborn / mpld3.

import base64
import json
import shutil
import subprocess

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sphinx

from matplotlib import image

logger = sphinx.util.logging.getLogger(__name__)

# Repo root: docs/sphinxext/generate_gallery.py -> repo
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOKS_ROOT = REPO_ROOT / "examples"

# A notebook's path under examples/ decides where it lands. A folder named here is a page:
# `examples/<page>/nb.ipynb` puts the notebook on that page with no section heading, and
# `examples/<page>/<section>/nb.ipynb` puts it in a named section of the page. Any other folder is a
# section of CATCH_ALL_PAGE, so `examples/<section>/nb.ipynb` lands there under that heading, and a
# notebook at the top level lands on CATCH_ALL_PAGE with no heading. Pages appear in the order listed
# here; a page with no notebooks writes no file at all.
GALLERY_PAGES = (
    ("getting_started", "Getting Started", "getting_started"),
    ("gallery", "Example Gallery", "gallery"),
)
CATCH_ALL_PAGE = "gallery"

# Sections are the folders themselves, titled by their folder name. List a folder here to pin where
# it sits on the page (anything unlisted follows, alphabetically), and give it a title only when
# title-casing the folder name would get it wrong.
SECTION_ORDER: tuple[str, ...] = ()
SECTION_TITLES: dict[str, str] = {"feed_forward": "Feed-forward"}

# Cell tag that picks the cell whose first image becomes the notebook's gallery thumbnail. Tag the same
# cell `remove-cell` to keep it off the rendered page; its output stays in the notebook for extraction.
THUMBNAIL_TAG = "gallery-thumbnail"

PAGE_TITLE = """
{title}
{underlines}
"""

TOCTREE_HEAD = """
.. toctree::
   :hidden:

"""

GRID_HEAD = """
.. grid:: 1 2 3 3
   :gutter: 4

"""

SECTION_TEMPLATE = """
.. _gallery-{section_id}:

{section_title}
{underlines}

.. grid:: 1 2 3 3
   :gutter: 4

"""

ITEM_TEMPLATE = """
   .. grid-item-card:: :doc:`{doc_name}`
      :img-top: {image}
      :link: {doc_reference}
      :link-type: {link_type}
      :shadow: none
"""


def is_tracked_by_git(filepath):
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(filepath)],
            capture_output=True,
            check=False,
            cwd=REPO_ROOT,
        )
    except FileNotFoundError:
        return True
    else:
        return result.returncode == 0


def _cell_images(cell):
    return [
        output["data"]["image/png"]
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]


def create_thumbnail(infile, width=275, height=275, border=4):
    im = image.imread(infile)
    rows, cols = im.shape[:2]
    size = max(rows, cols)
    # Pad the short side with white instead of cropping the long one, so no part of the figure (a
    # title, an axis label) is cut off however its saved bounding box came out.
    white = 1.0 if np.issubdtype(im.dtype, np.floating) else 255
    thumb = np.full((size, size, im.shape[2]), white, dtype=im.dtype)
    top, left = (size - rows) // 2, (size - cols) // 2
    thumb[top : top + rows, left : left + cols] = im
    thumb[:border, :, :3] = thumb[-border:, :, :3] = 0
    thumb[:, :border, :3] = thumb[:, -border:, :3] = 0

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], aspect="auto", frameon=False, xticks=[], yticks=[])
    ax.imshow(thumb, aspect="auto", resample=True, interpolation="bilinear")
    fig.savefig(infile, dpi=dpi)
    plt.close(fig)


class NotebookGenerator:
    """Extract a thumbnail and stage a notebook for inclusion in the gallery."""

    def __init__(self, src_nb: Path, category: str, examples_dir: Path, thumbnails_dir: Path):
        self.src_nb = src_nb
        self.stripped_name = src_nb.stem
        self.category = category
        self.staged_nb = examples_dir / category / f"{self.stripped_name}.ipynb"
        self.png_path = thumbnails_dir / category / f"{self.stripped_name}.png"

        with src_nb.open(encoding="utf-8") as fid:
            self.json_source = json.load(fid)

    def stage_notebook(self):
        self.staged_nb.parent.mkdir(parents=True, exist_ok=True)
        # Always re-copy: notebooks at the source can change between builds.
        shutil.copyfile(self.src_nb, self.staged_nb)

    def extract_preview_pic(self):
        """Return the first image of the cell tagged ``THUMBNAIL_TAG``, or else the last image."""
        cells = self.json_source["cells"]
        tagged_cells = [
            cell for cell in cells if THUMBNAIL_TAG in cell.get("metadata", {}).get("tags", [])
        ]
        if tagged_cells:
            tagged_pics = _cell_images(tagged_cells[0])
            if tagged_pics:
                return base64.b64decode(tagged_pics[0])
            logger.warning(
                f"The cell tagged {THUMBNAIL_TAG!r} in {self.src_nb.name} has no image output, "
                f"using the last image",
                type="thumbnail_extractor",
            )

        pics = [pic for cell in cells for pic in _cell_images(cell)]
        if pics:
            return base64.b64decode(pics[-1])
        return None

    def gen_previews(self):
        self.png_path.parent.mkdir(parents=True, exist_ok=True)
        if self.png_path.exists():
            logger.info(
                f"Custom thumbnail already exists for {self.src_nb.name}, skipping extraction",
                type="thumbnail_extractor",
            )
            return

        preview = self.extract_preview_pic()
        if preview is not None:
            with self.png_path.open("wb") as buff:
                buff.write(preview)
            create_thumbnail(self.png_path)
        else:
            logger.warning(
                f"No image found in {self.src_nb.name}; its gallery card will have no thumbnail. "
                f"Re-run the notebook with its outputs saved, or drop a PNG at {self.png_path}.",
                type="thumbnail_extractor",
            )


def discover_notebooks() -> dict[tuple[str, str | None], list[Path]]:
    """
    Group every notebook under ``examples/`` by the page and section its path puts it in.

    Returns
    -------
    grouped : dict mapping tuple to list of pathlib.Path
        Notebook paths keyed by ``(page, section)``, where ``section`` is None for a notebook that
        sits directly in a page's folder or at the top level of ``examples/``.
    """
    if not NOTEBOOKS_ROOT.exists():
        return {}

    known_pages = {page for page, _, _ in GALLERY_PAGES}
    grouped: dict[tuple[str, str | None], list[Path]] = {}
    for path in sorted(NOTEBOOKS_ROOT.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in path.parts:
            continue
        folders = path.relative_to(NOTEBOOKS_ROOT).parts[:-1]
        if folders and folders[0] in known_pages:
            page, section = folders[0], (folders[1] if len(folders) > 1 else None)
        else:
            page, section = CATCH_ALL_PAGE, (folders[0] if folders else None)
        grouped.setdefault((page, section), []).append(path)
    return grouped


def _section_title(section: str) -> str:
    return SECTION_TITLES.get(section, section.replace("_", " ").title())


def _page_layout(grouped):
    """
    Lay the discovered notebooks out into pages and the sections within them.

    Returns
    -------
    layout : list of tuple of str, str and list of tuple
        One entry per page that has notebooks, in the order the pages appear: the document name, its
        title, and its sections as ``(section title or None, notebook paths)``.
    """
    known_pages = {page for page, _, _ in GALLERY_PAGES}
    ordering = {section: index for index, section in enumerate(SECTION_ORDER)}

    def position(section: str | None) -> tuple[bool, int, str]:
        # Unsectioned notebooks lead the page; the rest follow SECTION_ORDER, then alphabetically.
        return section is not None, ordering.get(section, len(ordering)), section or ""

    layout = []
    for page, page_title, document in GALLERY_PAGES:
        # One entry per section, so two folders of the same name land under one heading rather than
        # two headings competing for the same label.
        merged: dict[str | None, list[Path]] = {}
        for (notebook_page, section), paths in grouped.items():
            on_this_page = notebook_page == page or (
                document == CATCH_ALL_PAGE and notebook_page not in known_pages
            )
            if on_this_page:
                merged.setdefault(section, []).extend(paths)

        if not merged:
            continue

        sections = [
            (None if section is None else _section_title(section), merged[section])
            for section in sorted(merged, key=position)
        ]
        layout.append((document, page_title, sections))
    return layout


def main(app):
    logger.info("Starting pytensor_ml gallery generation.")

    src_dir = Path(app.builder.srcdir)
    examples_dir = src_dir / "examples"
    thumbnails_dir = src_dir / "_thumbnails"
    examples_dir.mkdir(parents=True, exist_ok=True)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    grouped = discover_notebooks()
    if not grouped:
        logger.warning(
            "No notebooks found under examples/; no gallery pages will be written.",
            type="thumbnail_extractor",
        )

    for document, page_title, sections in _page_layout(grouped):
        toctree_entries: list[str] = []
        body: list[str] = []

        for section_title, nb_paths in sections:
            if section_title is None:
                body.append(GRID_HEAD)
            else:
                body.append(
                    SECTION_TEMPLATE.format(
                        section_title=section_title,
                        section_id=section_title.lower().replace(" ", "-"),
                        underlines="-" * len(section_title),
                    )
                )

            for nb_path in nb_paths:
                if not is_tracked_by_git(nb_path):
                    logger.info(
                        f"Skipping {nb_path.name}, not tracked by git",
                        type="thumbnail_extractor",
                    )
                    continue

                # Staged under the same folders it came from, so two notebooks sharing a name in
                # different sections cannot collide.
                relative_dir = nb_path.parent.relative_to(NOTEBOOKS_ROOT)
                nbg = NotebookGenerator(
                    src_nb=nb_path,
                    category=str(relative_dir) if relative_dir.parts else CATCH_ALL_PAGE,
                    examples_dir=examples_dir,
                    thumbnails_dir=thumbnails_dir,
                )
                nbg.stage_notebook()
                nbg.gen_previews()

                doc_name = f"{nbg.category}/{nbg.stripped_name}"
                toctree_entries.append(doc_name)
                # Relative to docs/source/ — the leading slash makes Sphinx resolve it from the source
                # root, so a hand-supplied thumbnail can be dropped in at the same path.
                img_path = f"/_thumbnails/{nbg.category}/{nbg.stripped_name}.png"
                body.append(
                    ITEM_TEMPLATE.format(
                        doc_name=doc_name,
                        image=img_path,
                        doc_reference=doc_name,
                        link_type="doc",
                    )
                )

        # Title, then a hidden toctree so the notebooks register with Sphinx, then the grid cards.
        file_lines = [
            PAGE_TITLE.format(title=page_title, underlines="=" * len(page_title)),
            TOCTREE_HEAD,
        ]
        file_lines.extend(f"   {entry}\n" for entry in toctree_entries)
        file_lines.append("\n")
        file_lines.extend(body)

        page_rst = examples_dir / f"{document}.rst"
        page_rst.write_text("\n".join(file_lines), encoding="utf-8")
        logger.info(f"Wrote {page_title} to {page_rst.relative_to(src_dir)}")


def setup(app):
    app.connect("builder-inited", main)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
