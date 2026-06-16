from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path.cwd()
MP_ROOT = ROOT / "mediapipe_"
DATA_DIR = next(p for p in MP_ROOT.iterdir() if p.is_dir())
LANDMARK_DIR = DATA_DIR / "landmark"
RESULT_DIR = DATA_DIR / "results"
LOG_DIR = RESULT_DIR / "training_logs"
OUT_DIR = MP_ROOT / "ppt_visuals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["side_lunge", "burpee", "plank", "pushup", "crunch"]
LABEL_DISPLAY = ["Side Lunge", "Burpee", "Plank", "Pushup", "Crunch"]
RED = "#be1423"
BLACK = "#222222"
GRAY = "#666666"
LIGHT = "#f4f4f4"


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def read_matrix(path: Path) -> np.ndarray:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return np.array([[int(float(v)) for v in row] for row in reader])


def make_frame_count_chart() -> Path:
    rows = []
    for path in sorted(LANDMARK_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        label = str(df["mode"].iloc[0]) if len(df) else path.stem.split("_")[0]
        rows.append({"label": label, "frames": len(df)})
    frame_df = pd.DataFrame(rows).sort_values("frames", ascending=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = [RED if label == "pushup" else "#222222" for label in frame_df["label"]]
    ax.bar(frame_df["label"], frame_df["frames"], color=colors)
    ax.set_title("Direct MediaPipe Capture: Frame Count by Exercise", fontsize=18, weight="bold", pad=16)
    ax.set_ylabel("Frames", fontsize=12)
    ax.set_xlabel("Exercise label", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(frame_df["frames"]):
        ax.text(i, v + max(frame_df["frames"]) * 0.02, f"{v:,}", ha="center", fontsize=11)
    note = "Input unit: frame-level MediaPipe CSV, not AI-Hub -3d.json sample-level input"
    ax.text(0.5, -0.2, note, transform=ax.transAxes, ha="center", fontsize=11, color=GRAY)
    out = OUT_DIR / "mediapipe_frame_counts.png"
    savefig(out)
    frame_df.to_csv(OUT_DIR / "mediapipe_frame_counts.csv", index=False, encoding="utf-8-sig")
    return out


def make_metric_bar() -> Path:
    summary = pd.read_csv(RESULT_DIR / "summary.csv")
    summary["model"] = summary["model"].replace({"xgb": "XGBoost", "svm": "SVM"})
    metric_cols = ["accuracy", "macro_f1", "weighted_f1"]

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    x = np.arange(len(summary))
    width = 0.22
    colors = ["#222222", RED, "#9a9a9a"]
    for idx, metric in enumerate(metric_cols):
        values = summary[metric].astype(float).values
        bars = ax.bar(x + (idx - 1) * width, values, width, label=metric.replace("_", "-"), color=colors[idx])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=10)
    ax.set_title("MediaPipe Preliminary Experiment: Model Metrics", fontsize=18, weight="bold", pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.5,
        -0.18,
        "Preliminary frame-level result. Do not directly compare with AI-Hub sample-level main experiment.",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color=GRAY,
    )
    out = OUT_DIR / "mediapipe_metrics_bar.png"
    savefig(out)
    return out


def make_results_table_image() -> Path:
    summary = pd.read_csv(RESULT_DIR / "summary.csv")
    summary["model"] = summary["model"].replace({"xgb": "XGBoost", "svm": "SVM"})
    display = summary[["model", "accuracy", "macro_f1", "weighted_f1"]].copy()
    for col in ["accuracy", "macro_f1", "weighted_f1"]:
        display[col] = display[col].astype(float).map(lambda v: f"{v:.4f}")
    display.columns = ["Model", "Accuracy", "Macro-F1", "Weighted-F1"]

    fig, ax = plt.subplots(figsize=(8.8, 2.5))
    ax.axis("off")
    ax.set_title("Result CSV Summary", fontsize=17, weight="bold", pad=12)
    tbl = ax.table(cellText=display.values, colLabels=display.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.65)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor(RED)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("white")
    out = OUT_DIR / "mediapipe_results_table.png"
    savefig(out)
    return out


def make_confusion_matrices() -> Path:
    matrices = [
        ("SVM", read_matrix(RESULT_DIR / "svm_confusion_matrix.csv")),
        ("XGBoost", read_matrix(RESULT_DIR / "xgb_confusion_matrix.csv")),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.7))
    for ax, (name, mat) in zip(axes, matrices):
        im = ax.imshow(mat, cmap="Reds")
        ax.set_title(f"{name} Confusion Matrix", fontsize=16, weight="bold", pad=12)
        ax.set_xticks(range(len(LABEL_DISPLAY)))
        ax.set_yticks(range(len(LABEL_DISPLAY)))
        ax.set_xticklabels(LABEL_DISPLAY, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(LABEL_DISPLAY, fontsize=9)
        ax.set_xlabel("Predicted label", fontsize=11)
        ax.set_ylabel("True label", fontsize=11)
        threshold = mat.max() * 0.55 if mat.max() else 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                color = "white" if mat[i, j] > threshold else BLACK
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=10, color=color)
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    fig.suptitle("MediaPipe Preliminary Experiment: Confusion Matrices", fontsize=19, weight="bold")
    fig.text(
        0.5,
        0.02,
        "The two matrices have different evaluation totals in the current result files, so they should be interpreted as preliminary outputs.",
        ha="center",
        fontsize=11,
        color=GRAY,
    )
    out = OUT_DIR / "mediapipe_confusion_matrices.png"
    savefig(out)
    return out


def make_training_log_visual() -> Path:
    rows = []
    for path in sorted(LOG_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        model = path.stem.replace("_log", "").replace("xgb", "XGBoost").replace("svm", "SVM")
        row = df.iloc[0].to_dict()
        row["Model"] = model
        rows.append(row)
    logs = pd.DataFrame(rows)
    display = logs[["Model", "accuracy", "macro_f1", "weighted_f1"]].copy()
    for col in ["accuracy", "macro_f1", "weighted_f1"]:
        display[col] = display[col].astype(float).map(lambda v: f"{v:.4f}")
    display.columns = ["Model", "Accuracy", "Macro-F1", "Weighted-F1"]

    fig, ax = plt.subplots(figsize=(8.8, 2.6))
    ax.axis("off")
    ax.set_title("Training Log CSV Snapshot", fontsize=17, weight="bold", pad=12)
    tbl = ax.table(cellText=display.values, colLabels=display.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.65)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor(BLACK)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("white")
    out = OUT_DIR / "mediapipe_training_log_table.png"
    savefig(out)
    return out


def make_screenshot_crop() -> Path:
    screenshots = sorted(p for p in MP_ROOT.glob("*.png") if p.is_file())
    if not screenshots:
        return Path()
    src = screenshots[0]
    image = Image.open(src).convert("RGB")
    w, h = image.size
    # Keep the app/result overlay and pose visualization, remove most of the media-player controls.
    crop = image.crop((0, 28, w, int(h * 0.90)))
    out = OUT_DIR / "mediapipe_capture_example_cropped.png"
    crop.save(out)
    return out


def write_index(paths: list[Path]) -> None:
    lines = [
        "# MediaPipe PPT Visual Artifacts",
        "",
        "These images visualize every CSV result file in `mediapipe_/Mediapipe 데이터/results` and summarize the directly captured MediaPipe landmark data.",
        "",
        "Important interpretation note:",
        "",
        "> 본 결과는 프레임 단위 예비 실험이며, AI-Hub 본실험의 샘플 단위 결과와 직접 비교하지 않는다.",
        "",
        "Generated files:",
    ]
    for path in paths:
        if path:
            lines.append(f"- `{path.name}`")
    (OUT_DIR / "README_MEDIAPIPE_PPT_VISUALS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    outputs = [
        make_screenshot_crop(),
        make_frame_count_chart(),
        make_metric_bar(),
        make_results_table_image(),
        make_confusion_matrices(),
        make_training_log_visual(),
    ]
    write_index(outputs)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
