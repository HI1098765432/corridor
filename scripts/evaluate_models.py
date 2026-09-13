"""Held-out segmentation evaluation.

The combined model was trained on all 71 labelled images, so there is no
held-out data for it and no honest generalisation number can be computed for
it from this material. Reporting its score on those same images would be
reporting how well it fits its own training set.

What the supplied data *does* support is a genuine cross-dataset test, because
two further models were trained on disjoint subsets:

    KK1Model  trained on KK1 (40 images)  ->  evaluated on KK2 (31 labelled)
    KK2Model  trained on KK2 (31 images)  ->  evaluated on KK1 (40 labelled)

Each of those pairings is a real held-out evaluation: the evaluation images
were never seen during that model's training. The combined model is also run
over everything, clearly labelled as a training-set fit, so the two numbers can
be compared without being confused for each other.

Instances are matched between prediction and ground truth by intersection over
union, one-to-one, using the same optimal assignment the tracker uses.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"

MODELS = {
    "KK1Model": TRAIN / "KK1Model" / "models" / "cyto2_phase_microfluidic_d10",
    "KK2Model": TRAIN / "KK2Model" / "models" / "cyto2_phase_microfluidic_d10",
    "combi": TRAIN / "KK1KK2_combiModel" / "models" / "cyto2_phase_microfluidic_KK1KK2_combi",
}
SETS = {"KK1": TRAIN / "KK1", "KK2": TRAIN / "KK2"}

#: (model, evaluation set) pairs where the evaluation images were never seen
#: during that model's training.
HELD_OUT = {("KK1Model", "KK2"), ("KK2Model", "KK1")}

IOU_THRESHOLDS = (0.5, 0.75, 0.9)


def load_pairs(directory: Path) -> list[tuple[Path, Path]]:
    """Image/label pairs. An image without a label is skipped, not assumed empty."""
    pairs = []
    for seg in sorted(directory.glob("*_seg.npy")):
        image = seg.with_name(seg.name.replace("_seg.npy", ".tif"))
        if image.exists():
            pairs.append((image, seg))
    return pairs


def ground_truth(seg_path: Path) -> np.ndarray:
    data = np.load(seg_path, allow_pickle=True).item()
    return np.asarray(data["masks"]).astype(np.int32)


def iou_matrix(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """IoU between every ground-truth and every predicted instance."""
    t_labels = [int(v) for v in np.unique(truth) if v]
    p_labels = [int(v) for v in np.unique(predicted) if v]
    if not t_labels or not p_labels:
        return np.zeros((len(t_labels), len(p_labels)), dtype=float)

    # A 2-D histogram of (truth label, predicted label) gives every pairwise
    # intersection in one pass, which matters at 70 images x 3 models.
    t_index = {lab: i for i, lab in enumerate(t_labels)}
    p_index = {lab: i for i, lab in enumerate(p_labels)}
    overlap = np.zeros((len(t_labels), len(p_labels)), dtype=np.int64)
    both = (truth > 0) & (predicted > 0)
    for t_val, p_val in zip(truth[both], predicted[both]):
        overlap[t_index[int(t_val)], p_index[int(p_val)]] += 1

    t_area = np.array([(truth == lab).sum() for lab in t_labels], dtype=np.int64)
    p_area = np.array([(predicted == lab).sum() for lab in p_labels], dtype=np.int64)
    union = t_area[:, None] + p_area[None, :] - overlap
    return np.where(union > 0, overlap / np.maximum(union, 1), 0.0)


@dataclass
class Tally:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ious: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mean_iou_of_matches": round(float(np.mean(self.ious)), 4) if self.ious else None,
        }


def score(truth: np.ndarray, predicted: np.ndarray, tallies: dict[float, Tally]) -> None:
    ious = iou_matrix(truth, predicted)
    n_truth, n_pred = ious.shape
    if n_truth and n_pred:
        rows, cols = linear_sum_assignment(-ious)  # one-to-one, maximising IoU
        matched = [(int(r), int(c), float(ious[r, c])) for r, c in zip(rows, cols)]
    else:
        matched = []
    for threshold, tally in tallies.items():
        hits = [m for m in matched if m[2] >= threshold]
        tally.tp += len(hits)
        tally.fp += n_pred - len(hits)
        tally.fn += n_truth - len(hits)
        tally.ious.extend(m[2] for m in hits)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cellprob", type=float, default=0.0)
    parser.add_argument("--flow", type=float, default=0.4)
    parser.add_argument("--limit", type=int, default=0, help="images per set, 0 = all")
    parser.add_argument("--out", default=str(ROOT / "docs" / "model_evaluation.json"))
    args = parser.parse_args()

    from cellpose import models

    datasets = {name: load_pairs(path) for name, path in SETS.items()}
    for name, pairs in datasets.items():
        print(f"{name}: {len(pairs)} labelled images")
        if args.limit:
            datasets[name] = pairs[: args.limit]

    report: dict = {
        "settings": {
            "cellprob_threshold": args.cellprob,
            "flow_threshold": args.flow,
            "channels": [0, 0],
            "note": (
                "These are the settings the researcher used when labelling, read "
                "from the _seg.npy files."
            ),
        },
        "evaluations": [],
    }

    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            print(f"  {model_name}: missing, skipped")
            continue
        model = models.CellposeModel(pretrained_model=str(model_path), gpu=False)
        for set_name, pairs in datasets.items():
            held_out = (model_name, set_name) in HELD_OUT
            tallies = {t: Tally() for t in IOU_THRESHOLDS}
            n_truth = n_pred = 0
            for image_path, seg_path in pairs:
                image = tifffile.imread(image_path)
                if image.ndim == 3 and image.shape[-1] in (3, 4):
                    image = image[..., 0]
                truth = ground_truth(seg_path)
                predicted = np.asarray(
                    model.eval(
                        image, channels=[0, 0], diameter=None,
                        cellprob_threshold=args.cellprob, flow_threshold=args.flow,
                    )[0]
                ).astype(np.int32)
                n_truth += len([v for v in np.unique(truth) if v])
                n_pred += len([v for v in np.unique(predicted) if v])
                score(truth, predicted, tallies)

            kind = "HELD OUT" if held_out else "training-set fit"
            entry = {
                "model": model_name,
                "evaluated_on": set_name,
                "held_out": held_out,
                "kind": kind,
                "n_images": len(pairs),
                "n_ground_truth_instances": n_truth,
                "n_predicted_instances": n_pred,
                "by_iou_threshold": {str(t): tallies[t].to_dict() for t in IOU_THRESHOLDS},
            }
            report["evaluations"].append(entry)
            main_tally = tallies[0.5]
            print(
                f"  {model_name:9s} on {set_name:3s} [{kind:16s}] "
                f"n={len(pairs):>2} gt={n_truth:>3} pred={n_pred:>3} | "
                f"IoU>=0.5  P={main_tally.precision:.3f} R={main_tally.recall:.3f} "
                f"F1={main_tally.f1:.3f} "
                f"mIoU={np.mean(main_tally.ious):.3f}" if main_tally.ious else ""
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
