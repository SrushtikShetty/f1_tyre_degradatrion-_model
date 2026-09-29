import os
import re
import joblib
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_FILE = "f1_tyre_wear_model.joblib"
MISSING_VALUE = -999.0


# ============================================================================
# HELPERS
# ============================================================================

def ask_number(label, default=None, allow_blank=True):
    """Ask for a numeric value."""
    while True:
        suffix = ""
        if default is not None:
            suffix = f" [{default}]"
        elif allow_blank:
            suffix = " [blank = missing]"

        answer = input(f"{label}{suffix}: ").strip()

        if answer == "":
            if default is not None:
                return float(default)
            if allow_blank:
                return MISSING_VALUE

        try:
            value = float(answer)
        except ValueError:
            print("Please enter a numeric value.")
            continue

        if not np.isfinite(value):
            print("Please enter a finite numeric value.")
            continue

        return value


def ask_choice(label, options):
    """Ask the user to choose one categorical value."""
    options = list(dict.fromkeys(options))

    print(f"\n{label}")
    for i, option in enumerate(options, start=1):
        print(f"  {i}. {option}")

    while True:
        answer = input("Choose number: ").strip()

        try:
            index = int(answer)
            if 1 <= index <= len(options):
                return options[index - 1]
        except ValueError:
            pass

        print(f"Please enter a number from 1 to {len(options)}.")


def sanitize_name(name):
    """Match the sanitization used in train_model.py."""
    return re.sub(r"[\[\]<>]", "_", str(name))


def get_dummy_options(feature_names, categorical_column):
    """
    Recover category names from the one-hot encoded feature names saved
    inside the trained model.

    Example:
        weather_Rain
        weather_Sunny
        weather_UNKNOWN
    """
    prefix = sanitize_name(categorical_column) + "_"
    options = []

    for feature in feature_names:
        if feature.startswith(prefix):
            category = feature[len(prefix):]

            # Ignore pandas dummy-na artifacts when possible.
            if category and category != "nan":
                options.append(category)

    return options


def build_input_row(artifact):
    """
    Interactively collect the features expected by the trained model.

    Important:
    The training script one-hot encodes categorical variables and removes
    deterministic target features. Therefore this function constructs the
    final encoded feature vector exactly in artifact['feature_columns'] order.
    """
    feature_names = artifact["feature_columns"]
    categorical_columns = artifact.get("categorical_columns", [])

    print("\n" + "=" * 75)
    print("INTERACTIVE PREDICTION V2")
    print("=" * 75)

    print(f"\nModel expects {len(feature_names)} final encoded features.")
    print("You will be asked for the available model inputs.")
    print("Press ENTER on numeric fields to use the missing-value sentinel.")
    print("\nIMPORTANT: Do not enter future-lap information.")

    # ------------------------------------------------------------------------
    # Build categorical selections first.
    # ------------------------------------------------------------------------

    selected_categories = {}

    for category in categorical_columns:
        options = get_dummy_options(feature_names, category)

        if options:
            selected_categories[category] = ask_choice(
                f"{category} ({len(options)} known categories)",
                options,
            )

    # ------------------------------------------------------------------------
    # Build final row.
    #
    # Start everything at the same sentinel used during training.
    # Then replace numeric/categorical values supplied by the user.
    # ------------------------------------------------------------------------

    row = {feature: MISSING_VALUE for feature in feature_names}

    # Set one-hot categorical features.
    for category, selected in selected_categories.items():
        prefix = sanitize_name(category) + "_"
        encoded_name = prefix + sanitize_name(selected)

        # Handle the unlikely case where sanitization changed the name.
        if encoded_name in row:
            row[encoded_name] = 1.0

    # ------------------------------------------------------------------------
    # Numeric features.
    #
    # A feature is treated as numeric if it isn't one of the known
    # categorical dummy columns.
    # ------------------------------------------------------------------------

    categorical_dummy_features = set()

    for category in categorical_columns:
        prefix = sanitize_name(category) + "_"
        for feature in feature_names:
            if feature.startswith(prefix):
                categorical_dummy_features.add(feature)

    numeric_features = [
        feature
        for feature in feature_names
        if feature not in categorical_dummy_features
    ]

    print("\n" + "-" * 75)
    print(f"NUMERIC MODEL FEATURES ({len(numeric_features)})")
    print("-" * 75)
    print(
        "Enter as many real/current-history values as you have. "
        "Blank values remain -999, exactly as during training."
    )

    for feature in numeric_features:
        value = ask_number(feature)
        row[feature] = value

    # Final DataFrame in EXACT training order.
    X_input = pd.DataFrame(
        [[row[feature] for feature in feature_names]],
        columns=feature_names,
    ).astype(np.float32)

    return X_input


def print_feature_summary(X_input):
    """Show how complete the interactive input was."""
    values = X_input.iloc[0]

    supplied = int((values != MISSING_VALUE).sum())
    total = len(values)
    missing = total - supplied

    print("\n" + "=" * 75)
    print("INPUT SUMMARY")
    print("=" * 75)
    print(f"Model features supplied : {supplied}/{total}")
    print(f"Model features missing  : {missing}/{total}")

    if total:
        print(f"Input completeness      : {supplied / total * 100:.1f}%")

    if missing > 0:
        print(
            "\nSome model features are still using -999. "
            "A prediction can still be produced, but a more complete "
            "history/context input is preferable."
        )
    else:
        print("\nAll model features were supplied.")


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
