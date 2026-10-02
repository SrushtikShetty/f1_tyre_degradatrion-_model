from __future__ import annotations

from inference.interactive_predict import main


if __name__ == "__main__":
    main()


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Could not find {MODEL_FILE}. "
            f"Run train_model.py first."
        )

    artifact = joblib.load(MODEL_FILE)

    if "model" not in artifact:
        raise KeyError("The model artifact does not contain 'model'.")

    if "feature_columns" not in artifact:
        raise KeyError(
            "The model artifact does not contain 'feature_columns'."
        )

    model = artifact["model"]
    feature_names = artifact["feature_columns"]

    print("\n" + "=" * 75)
    print("F1 TYRE DEGRADATION PREDICTOR - INTERACTIVE V2")
    print("=" * 75)

    print(f"\nModel file : {MODEL_FILE}")
    print(f"Features   : {len(feature_names)}")
    print(f"Device     : {artifact.get('device', 'unknown')}")

    X_input = build_input_row(artifact)

    print_feature_summary(X_input)

    # ------------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------------

    print("\n" + "=" * 75)
    print("RUNNING PREDICTION")
    print("=" * 75)

    # We intentionally keep the CPU/GPU handling simple.
    # The model can accept the NumPy input; XGBoost may issue its normal
    # CPU/GPU mismatch performance warning if the model was trained on CUDA.
    prediction = float(model.predict(X_input.to_numpy())[0])

    metrics = artifact.get("oof_metrics", {})

    print("\nPREDICTION")
    print("-" * 75)
    print(f"Predicted next-lap degradation : {prediction:.6f}%")

    print("\nVALIDATED MODEL METRICS")
    print("-" * 75)
    print(f"Δwear R²   : {metrics.get('delta_r2', float('nan')):.6f}")
    print(f"Δwear MAE  : {metrics.get('delta_mae', float('nan')):.6f}")
    print(f"Δwear RMSE : {metrics.get('delta_rmse', float('nan')):.6f}")

    print("\nModel target:")
    print("  next_lap_wear_increment = predicted additional tyre wear")

    print("\nNote:")
    print(
        "The training script reconstructs absolute next-lap tyre wear as "
        "current observed wear + predicted Δwear. This interactive model "
        "predicts Δwear only because current tyre wear was intentionally "
        "excluded from the feature matrix."
    )


if __name__ == "__main__":
    main()
