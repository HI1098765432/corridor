"""Measure which modules Corridor's segmentation path actually loads.

The 1.x spec shipped ``collect_submodules("cellpose")``: every Cellpose module,
including its Qt GUI, its training loop and its Dask ``contrib`` tree, none of
which a user's Analyse ever touches -- and whose optional imports dragged their
own dependencies into the bundle through PyInstaller's static analysis. The
replacement is a list, and a list is only as good as the evidence behind it, so
this script produces the evidence rather than the list being written by hand.

What it does, in a *fresh* interpreter (the parent's ``sys.modules`` already
holds whatever this script imported, so measuring in-process would report this
script, not the runtime):

1. ``from cellpose import models`` -- the import ``SegmentationService._load``
   makes.
2. ``CellposeModel(pretrained_model=<absolute path>, gpu=False)`` with the
   production model, the only way the application constructs one.
3. ``model.eval`` on a synthetic 128 x 128 float frame with ``channels=[0, 0]``,
   once with the default normalisation and once with a tiled + sharpened one,
   because the normalisation presets are the only switch the application has
   that takes ``eval`` down a different code path.

It then writes, beside the spec:

* ``packaging/cellpose_runtime_modules.txt`` -- the sorted ``cellpose*``
  modules that were loaded. The spec ships exactly these as hidden imports, and
  ``tests/test_packaging.py`` fails if the spec stops covering them.
* ``packaging/runtime_imports.json`` -- provenance (versions, model hash), every
  third-party top-level package the path loaded, and a static scan of
  ``src/corridor`` for imports of the packages the spec may exclude. The spec
  refuses to exclude anything this file shows as loaded, and refuses to build
  against a Cellpose version other than the one measured.

Run it again whenever Cellpose, torch or the production model changes::

    python scripts/measure_runtime_imports.py --model <path to the production model>

It imports torch and loads the model (about 1 GB of memory), so it caps the
thread pools at two by default; pass ``--threads`` to change that.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"
APP_SOURCE = ROOT / "src" / "corridor"
MODULES_FILE = "cellpose_runtime_modules.txt"
IMPORTS_FILE = "runtime_imports.json"

#: Packages the spec may exclude, so their use by the application's own source
#: is recorded alongside the runtime measurement. Static, so it also sees the
#: imports that sit inside functions (napari's are deliberately lazy).
SCANNED_PACKAGES = ("pandas", "napari", "dask", "torchvision", "matplotlib", "sklearn")

#: The thread-pool variables torch, OpenMP, MKL and Numba read at start-up.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def scan_app_imports(source: Path = APP_SOURCE) -> dict[str, list[str]]:
    """``package -> ["relative/path.py:line", ...]`` for import statements."""
    pattern = re.compile(
        r"^\s*(?:import|from)\s+(" + "|".join(SCANNED_PACKAGES) + r")\b", re.MULTILINE
    )
    found: dict[str, list[str]] = {name: [] for name in SCANNED_PACKAGES}
    for path in sorted(source.rglob("*.py")):
        text = path.read_text(encoding="utf-8-sig")
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            rel = path.relative_to(source.parent).as_posix()
            found[match.group(1)].append(f"{rel}:{line}")
    return found


# --------------------------------------------------------------------------
# Child: the measurement itself
# --------------------------------------------------------------------------


def _synthetic_frame():
    """Elongated bright bodies on a noisy background, deterministic.

    The content only has to make ``eval`` run (network, flow dynamics, mask
    construction); whether it finds the bodies is recorded but is not the
    point. With Cellpose 3.1.1.3 and the production model it finds none
    (measured 2026-10-02). That does not narrow the module list: in that
    release the only imports inside functions of the loaded modules are
    ``models -> segformer`` (transformer backbones only), ``utils -> ssl``
    (model download) and ``io -> models`` (already loaded); every other import
    runs when the module loads. Re-check that after a Cellpose upgrade.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    frame = 0.05 * rng.standard_normal((128, 128)).astype(np.float32)
    for cy, cx in ((32.0, 30.0), (70.0, 64.0), (96.0, 100.0)):
        frame += np.exp(-(((xx - cx) / 5.0) ** 2 + ((yy - cy) / 12.0) ** 2) / 2.0)
    return frame.astype(np.float32)


def _child(model: Path, out: Path) -> int:
    baseline = set(sys.modules)
    started = time.perf_counter()

    from cellpose import models  # the import SegmentationService._load makes

    import cellpose
    import torch

    net = models.CellposeModel(pretrained_model=str(model), gpu=False)
    frame = _synthetic_frame()
    calls = []
    for label, normalize in (
        ("default", True),
        # The ``local_sharpen`` preset's dict, as SegmentationConfig builds it.
        ("tile_norm_blocksize=128, sharpen_radius=15",
         {"tile_norm_blocksize": 128, "sharpen_radius": 15}),
    ):
        masks = net.eval(
            frame,
            channels=[0, 0],
            diameter=None,
            cellprob_threshold=0.0,
            flow_threshold=0.4,
            normalize=normalize,
        )[0]
        calls.append({"normalize": label, "instances_found": int(masks.max())})

    loaded = set(sys.modules) - baseline
    top_level = sorted({name.split(".")[0] for name in loaded})
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    third_party = [name for name in top_level if name not in stdlib]

    try:
        import importlib.metadata as md

        owners = md.packages_distributions()
        distributions = sorted(
            {dist for name in third_party for dist in owners.get(name, [])}
        )
    except Exception:  # noqa: BLE001 - provenance only
        distributions = []

    record = {
        "cellpose_version": str(cellpose.version),
        "torch_version": str(torch.__version__),
        "torch_threads": int(torch.get_num_threads()),
        "eval_calls": calls,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "cellpose_modules": sorted(n for n in loaded if n.split(".")[0] == "cellpose"),
        "third_party_top_level": third_party,
        "distributions": distributions,
    }
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------
# Parent: run the child, then write the packaging files
# --------------------------------------------------------------------------


def _modules_text(record: dict) -> str:
    model = record["model"]
    lines = [
        "# Cellpose modules loaded by Corridor's segmentation path. MEASURED, not",
        "# written by hand: regenerate with",
        "#   python scripts/measure_runtime_imports.py --model <production model>",
        f"# cellpose {record['cellpose_version']}, torch {record['torch_version']}, "
        f"Python {record['python']}, {record['platform']}",
        f"# model {model['filename']} sha256 {model['sha256']}",
        f"# measured {record['measured_utc']}; provenance in runtime_imports.json",
        "# packaging/corridor.spec ships exactly these as hidden imports, and",
        "# tests/test_packaging.py fails if its list stops covering them.",
    ]
    lines += record["cellpose_modules"]
    return "\n".join(lines) + "\n"


def measure(model: Path, *, threads: str, out_dir: Path) -> dict:
    env = dict(os.environ)
    for variable in THREAD_VARIABLES:
        env[variable] = threads
    with tempfile.TemporaryDirectory() as scratch:
        result = Path(scratch) / "measurement.json"
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--child", "--model", str(model), "--json", str(result),
        ]
        completed = subprocess.run(command, env=env)
        if completed.returncode != 0 or not result.exists():
            raise SystemExit(
                f"The measurement process failed (exit code {completed.returncode})."
            )
        record = json.loads(result.read_text(encoding="utf-8"))

    record = {
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "thread_cap": threads,
        # The file name and hash, never the absolute path: this file is
        # committed, and where a developer keeps the weights is not part of it.
        "model": {"filename": model.name, "sha256": sha256_file(model)},
        **record,
        "app_source_imports": scan_app_imports(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MODULES_FILE).write_text(_modules_text(record), encoding="utf-8")
    (out_dir / IMPORTS_FILE).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", type=Path, required=True,
        help="absolute path to the production Cellpose model file",
    )
    parser.add_argument("--threads", default="2", help="thread-pool cap (default 2)")
    parser.add_argument("--out-dir", type=Path, default=PACKAGING)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    model = args.model.resolve()
    if not model.is_file():
        raise SystemExit(f"Model file not found: {model}")
    if args.child:
        return _child(model, args.json)

    record = measure(model, threads=args.threads, out_dir=args.out_dir)
    print(f"cellpose {record['cellpose_version']}, torch {record['torch_version']} "
          f"({record['torch_threads']} threads), {record['elapsed_seconds']} s")
    for call in record["eval_calls"]:
        print(f"  eval normalize={call['normalize']}: "
              f"{call['instances_found']} instances")
    print(f"{len(record['cellpose_modules'])} cellpose modules loaded:")
    for name in record["cellpose_modules"]:
        print(f"  {name}")
    print("third-party top-level packages loaded:")
    print("  " + ", ".join(record["third_party_top_level"]))
    for package, sites in record["app_source_imports"].items():
        print(f"src/corridor imports {package}: {', '.join(sites) or 'nowhere'}")
    print(f"wrote {args.out_dir / MODULES_FILE}")
    print(f"wrote {args.out_dir / IMPORTS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
