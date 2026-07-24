"""Fine-tune a nano YOLO detector off-device and optionally export NCNN."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Haptos obstacle detector on a development computer")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML path")
    parser.add_argument("--base-model", default="yolov8n.pt", help="Nano checkpoint used to initialize training")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size")
    parser.add_argument("--project", default="runs/haptos", help="Training output directory")
    parser.add_argument("--name", default="obstacle_nano", help="Training run name")
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience")
    parser.add_argument("--device", default=None, help="Training device, for example 0 or cpu")
    parser.add_argument("--export-ncnn", action="store_true", help="Export best.pt to NCNN after training")
    parser.add_argument("--export-imgsz", type=int, default=512, help="NCNN inference image size")
    args = parser.parse_args()

    if not Path(args.data).exists():
        parser.error(f"dataset YAML does not exist: {args.data}")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.imgsz <= 0 or args.export_imgsz <= 0:
        parser.error("--imgsz and --export-imgsz must be positive")
    if args.batch <= 0:
        parser.error("--batch must be positive")
    if args.patience < 0:
        parser.error("--patience must be 0 or greater")
    return args


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc

    model = YOLO(args.base_model)
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)
    best_path = Path(model.trainer.best)
    print(f"Best checkpoint: {best_path}")

    if args.export_ncnn:
        exported = YOLO(str(best_path)).export(format="ncnn", imgsz=args.export_imgsz)
        print(f"NCNN export: {exported}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
