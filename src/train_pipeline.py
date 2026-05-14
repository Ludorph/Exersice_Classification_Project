from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from data_utils import (
    CLASSES,
    build_sequence_dataset,
    build_tabular_dataset,
    load_labels,
    make_label_encoder,
    make_splits,
    sequence_to_graph_tensor,
)
from evaluation import evaluate_predictions
from models import LSTMClassifier, STGCNClassifier


DEFAULT_DATA_DIR = Path(r"C:\Users\Ludorph\Downloads\Physical Exercise RecognitionTime Series Dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare XGBoost, LSTM, and ST-GCN for exercise classification.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-analysis", action="store_true", help="Do not run analyze_results.py after training.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_xgboost(tabular, num_classes: int, seed: int) -> XGBClassifier:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(
        tabular.x_train,
        tabular.y_train,
        eval_set=[(tabular.x_val, tabular.y_val)],
        verbose=False,
    )
    return model


def save_predictions(
    output_dir: Path,
    model_name: str,
    test_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> None:
    rows = pd.DataFrame(
        {
            "vid_id": test_ids,
            "true_label": [class_names[i] for i in y_true],
            "pred_label": [class_names[i] for i in y_pred],
            "is_correct": y_true == y_pred,
        }
    )
    rows.to_csv(output_dir / f"test_predictions_{model_name}.csv", index=False, encoding="utf-8-sig")


def save_xgboost_importance(output_dir: Path, model: XGBClassifier, feature_names: list[str]) -> None:
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "xgboost_feature_importance.csv", index=False, encoding="utf-8-sig")


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def class_weight_tensor(y: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_torch_classifier(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> np.ndarray:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(y_train, num_classes, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)

    best_val_loss = float("inf")
    wait = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(xb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, checkpoint_path)
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is None:
        best_state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(best_state)
    model.eval()

    test_loader = make_loader(x_test, np.zeros(len(x_test), dtype=np.int64), args.batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in test_loader:
            logits = model(xb.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.data_dir, CLASSES)
    encoder = make_label_encoder(labels)
    class_names = list(encoder.classes_)
    splits = make_splits(labels, seed=args.seed)

    split_info = {
        "train": len(splits.train),
        "val": len(splits.val),
        "test": len(splits.test),
        "classes": class_names,
        "label_counts": labels["class"].value_counts().to_dict(),
    }
    (output_dir / "split_info.json").write_text(json.dumps(split_info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(split_info, indent=2, ensure_ascii=False))

    results: list[dict] = []

    print("\n[1/3] XGBoost")
    tabular = build_tabular_dataset(args.data_dir, labels, splits, encoder)
    xgb_model = train_xgboost(tabular, len(class_names), args.seed)
    xgb_pred = xgb_model.predict(tabular.x_test)
    save_predictions(output_dir, "xgboost", splits.test, tabular.y_test, xgb_pred, class_names)
    save_xgboost_importance(output_dir, xgb_model, tabular.feature_names)
    results.append(evaluate_predictions(tabular.y_test, xgb_pred, class_names, "xgboost", output_dir))

    print("\n[2/3] LSTM")
    sequence = build_sequence_dataset(args.data_dir, labels, splits, encoder, seq_len=args.seq_len)
    lstm = LSTMClassifier(input_size=sequence.x_train.shape[-1], num_classes=len(class_names))
    lstm_pred = train_torch_classifier(
        lstm,
        sequence.x_train,
        sequence.y_train,
        sequence.x_val,
        sequence.y_val,
        sequence.x_test,
        len(class_names),
        args,
        device,
        output_dir / "checkpoints" / "lstm.pt",
    )
    save_predictions(output_dir, "lstm", splits.test, sequence.y_test, lstm_pred, class_names)
    results.append(evaluate_predictions(sequence.y_test, lstm_pred, class_names, "lstm", output_dir))

    print("\n[3/3] ST-GCN")
    stgcn_x_train = sequence_to_graph_tensor(sequence.x_train)
    stgcn_x_val = sequence_to_graph_tensor(sequence.x_val)
    stgcn_x_test = sequence_to_graph_tensor(sequence.x_test)
    stgcn = STGCNClassifier(num_classes=len(class_names))
    stgcn_pred = train_torch_classifier(
        stgcn,
        stgcn_x_train,
        sequence.y_train,
        stgcn_x_val,
        sequence.y_val,
        stgcn_x_test,
        len(class_names),
        args,
        device,
        output_dir / "checkpoints" / "stgcn.pt",
    )
    save_predictions(output_dir, "stgcn", splits.test, sequence.y_test, stgcn_pred, class_names)
    results.append(evaluate_predictions(sequence.y_test, stgcn_pred, class_names, "stgcn", output_dir))

    summary = pd.DataFrame(results)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Metrics Summary ===")
    print(summary.to_string(index=False))

    if not args.skip_analysis:
        analysis_script = Path(__file__).with_name("analyze_results.py")
        subprocess.run(
            [sys.executable, str(analysis_script), "--output-dir", str(output_dir)],
            check=True,
        )


if __name__ == "__main__":
    main()
