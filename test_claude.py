from backend.models.risk.lgbm_risk import predict_single, load_model
from backend.models.risk.calibration import load_calibrator
from backend.models.risk.shap_explainer import load_explainer, explain_from_dict
from backend.models.risk.lgbm_risk import load_model

model, encoders, feature_cols = load_model()
calibrator = load_calibrator()
explainer = load_explainer()

# Test 1: raw prediction
raw = predict_single(
    zone_id=1, hour=18, day_of_week=4, month=1,
    police_station="UNKNOWN", vehicle_type="UNKNOWN",
    primary_violation_type="UNKNOWN",
    rolling_7d_count=50.0, rolling_30d_count=40.0,
    model=model, encoders=encoders, feature_cols=feature_cols
)
print(f"Raw predicted violations: {raw:.2f}")

# Test 2: calibrated risk score
import numpy as np
score = calibrator.predict_score(np.array([raw]))[0]
print(f"Calibrated risk score: {score:.1f}/100  ({calibrator.risk_label(score)})")

# Test 3: SHAP explanation
explanation = explain_from_dict(
    {"hour": 18, "is_rush_hour": 1, "rolling_7d_count": 50, "day_of_week": 4},
    feature_cols, explainer, top_n=5
)
for e in explanation:
    print(f"  {e['direction']} {e['feature']}: impact={e['impact']:.3f}")