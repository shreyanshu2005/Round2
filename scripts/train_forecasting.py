"""
train_forecasting.py
---------------------
End-to-end training script for Layer 5: Prophet + LSTM.

Usage:
    python scripts/train_forecasting.py [--skip-prophet] [--skip-lstm] [--epochs N]

Order:
    1. Verify feature_store.parquet exists (prerequisite from Layer 2)
    2. Train per-junction Prophet models + global fallback
    3. Train LSTM on top-20 junctions
    4. Print summary report
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_forecasting")

FEATURE_STORE = Path("data/processed/feature_store.parquet")

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train BTIP forecasting models (Layer 5)")
    parser.add_argument("--skip-prophet", action="store_true", help="Skip Prophet training")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM training")
    parser.add_argument("--epochs", type=int, default=30, help="LSTM training epochs")
    parser.add_argument(
        "--feature-store",
        default=str(FEATURE_STORE),
        help="Path to feature_store.parquet",
    )
    args = parser.parse_args()

    if not Path(args.feature_store).exists():
        logger.error(f"Feature store not found at {args.feature_store}")
        logger.error("Run Layer 2 preprocessing first: python scripts/build_feature_store.py")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("BTIP Layer 5 — Forecasting Model Training")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Prophet
    # -----------------------------------------------------------------------
    if not args.skip_prophet:
        logger.info("\n[1/2] Training Prophet models (per-junction + global fallback)...")
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from backend.models.forecasting.prophet_forecast import train_all

            prophet_summary = train_all(feature_store_path=args.feature_store)
            logger.info(
                f"Prophet complete: "
                f"{prophet_summary['trained_junctions']} junctions trained, "
                f"{prophet_summary['skipped_junctions']} skipped (sparse), "
                f"global_fallback={prophet_summary['global_model']}"
            )
        except ImportError:
            logger.error("prophet package not installed. Run: pip install prophet")
            logger.error("Skipping Prophet training.")
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            logger.error("Continuing to LSTM...")
    else:
        logger.info("[1/2] Skipping Prophet (--skip-prophet)")

    # -----------------------------------------------------------------------
    # 2. LSTM
    # -----------------------------------------------------------------------
    if not args.skip_lstm:
        logger.info("\n[2/2] Training LSTM on top-20 junctions...")
        try:
            from backend.models.forecasting.lstm_forecast import train

            lstm_summary = train(
                feature_store_path=args.feature_store,
                horizon_hours=168,  # 7d — covers both 24h and 7d endpoints
                epochs=args.epochs,
            )
            if lstm_summary.get("trained"):
                logger.info(
                    f"LSTM complete: "
                    f"best_val_loss={lstm_summary['best_val_loss']:.4f}, "
                    f"samples={lstm_summary['samples_used']}, "
                    f"top20={lstm_summary['top20_junctions']}"
                )
            else:
                logger.warning(f"LSTM did not train: {lstm_summary.get('reason')}")
        except ImportError:
            logger.error("PyTorch not installed. Run: pip install torch")
            logger.error("Skipping LSTM training.")
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
    else:
        logger.info("[2/2] Skipping LSTM (--skip-lstm)")

    # -----------------------------------------------------------------------
    # 3. Verification
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Verification")
    logger.info("=" * 60)

    from pathlib import Path as P
    artifacts = {
        "Prophet global model": P("models/saved/prophet/global_model.pkl"),
        "LSTM model weights": P("models/saved/lstm/lstm_model.pt"),
        "LSTM scaler": P("models/saved/lstm/scaler.pkl"),
        "LSTM top-20 list": P("models/saved/lstm/top20_junctions.pkl"),
    }
    for name, path in artifacts.items():
        status = "✓" if path.exists() else "✗ MISSING"
        logger.info(f"  {status}  {name}: {path}")

    prophet_dir = P("models/saved/prophet")
    junction_models = list(prophet_dir.glob("junction_*.pkl")) if prophet_dir.exists() else []
    logger.info(f"  ✓  Prophet junction models: {len(junction_models)}")

    logger.info("\nLayer 5 training complete.")


if __name__ == "__main__":
    main()