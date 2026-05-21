"""
FairLens – FastAPI Backend (Pure Python - No Pandas/Numpy CPU crashes)
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RAW_DATA = [
    {"Income": 85, "Term": 36, "Gender": "M", "True Label": 1},
    {"Income": 90, "Term": 24, "Gender": "M", "True Label": 1},
    {"Income": 45, "Term": 60, "Gender": "M", "True Label": 0},
    {"Income": 50, "Term": 48, "Gender": "M", "True Label": 0},
    {"Income": 88, "Term": 36, "Gender": "M", "True Label": 1},
    {"Income": 42, "Term": 60, "Gender": "F", "True Label": 0},
    {"Income": 55, "Term": 48, "Gender": "F", "True Label": 0},
    {"Income": 80, "Term": 36, "Gender": "F", "True Label": 1},
    {"Income": 62, "Term": 24, "Gender": "F", "True Label": 1},
    {"Income": 48, "Term": 60, "Gender": "F", "True Label": 0},
]

THRESHOLD_BIASED = 28
THRESHOLD_FAIR   = 25

def _base_score(income: float, term: int) -> float:
    term_factor = 1.0 - (term - 24) / 100
    return income * 0.8 * term_factor

def _biased_pred(income: float, term: int, gender: str) -> int:
    score = _base_score(income, term)
    if gender == "F":
        score -= 12
    return 1 if score > THRESHOLD_BIASED else 0

def _fair_pred(income: float, term: int) -> int:
    score = _base_score(income, term)
    return 1 if score > THRESHOLD_FAIR else 0

def _build_dataset() -> List[dict]:
    dataset = []
    for r in RAW_DATA:
        row = r.copy()
        row["biased_pred"] = _biased_pred(row["Income"], row["Term"], row["Gender"])
        row["true_label"] = row.pop("True Label")
        dataset.append(row)
    return dataset

class DatasetPayload(BaseModel):
    dataset: List[dict]

@app.get("/api/dataset")
def get_dataset():
    return {"dataset": _build_dataset()}

@app.post("/api/detect-bias")
def detect_bias(payload: DatasetPayload):
    dataset = payload.dataset
    
    if not dataset:
        raise HTTPException(status_code=422, detail="Dataset is empty")
        
    males = [r for r in dataset if r.get("Gender") == "M"]
    females = [r for r in dataset if r.get("Gender") == "F"]

    if not males or not females:
        raise HTTPException(status_code=422, detail="Dataset must contain both male and female applicants.")

    approval_rate_male = sum(r.get("biased_pred", 0) for r in males) / len(males)
    approval_rate_female = sum(r.get("biased_pred", 0) for r in females) / len(females)

    if approval_rate_male == 0:
        di_score = 1.0
    else:
        di_score = float(approval_rate_female / approval_rate_male)

    if di_score >= 0.9:
        verdict = "fair"
        message = "The model treats all genders equally (DI ≥ 0.9 — IEEE 7003-2024 compliant)."
    elif di_score <= 0.8:
        verdict = "biased"
        message = "Bias detected: female applicants are disadvantaged (DI ≤ 0.8 — IEEE 7003-2024 violation)."
    else:
        verdict = "marginal"
        message = "Marginal fairness (0.8 < DI < 0.9). Mitigation recommended."

    return {
        "di_score": round(di_score, 4),
        "approval_rate_male": round(approval_rate_male, 4),
        "approval_rate_female": round(approval_rate_female, 4),
        "verdict": verdict,
        "message": message,
        "threshold_fair": 0.9,
        "threshold_biased": 0.8,
    }

@app.post("/api/mitigate")
def mitigate(payload: DatasetPayload):
    dataset = payload.dataset
    cf_details = []

    for row in dataset:
        income = int(row["Income"])
        term   = int(row["Term"])
        gender = str(row["Gender"])

        fp = _fair_pred(income, term)

        cf_gender     = "F" if gender == "M" else "M"
        original_bp   = _biased_pred(income, term, gender)
        cf_bp         = _biased_pred(income, term, cf_gender)
        gender_driven = (original_bp != cf_bp)

        row["fair_pred"] = fp
        cf_details.append({
            "original_biased_pred": original_bp,
            "cf_biased_pred":       cf_bp,
            "gender_was_decisive":  gender_driven,
            "fair_pred":            fp,
        })

    males = [r for r in dataset if r.get("Gender") == "M"]
    females = [r for r in dataset if r.get("Gender") == "F"]

    approval_rate_male = sum(r.get("fair_pred", 0) for r in males) / len(males) if males else 0.0
    approval_rate_female = sum(r.get("fair_pred", 0) for r in females) / len(females) if females else 0.0

    if approval_rate_male == 0:
        fair_di = 1.0
    else:
        fair_di = float(approval_rate_female / approval_rate_male)

    correct_preds = sum(1 for r in dataset if r.get("fair_pred") == r.get("true_label"))
    accuracy = correct_preds / len(dataset) if dataset else 0.0

    return {
        "dataset": dataset,
        "fair_di_score": round(fair_di, 4),
        "approval_rate_male": round(approval_rate_male, 4),
        "approval_rate_female": round(approval_rate_female, 4),
        "accuracy": round(accuracy, 4),
        "cf_details": cf_details,
        "verdict": "fair" if fair_di >= 0.9 else "marginal",
        "message": (
            "Counterfactual fairness applied — gender penalty removed. "
            f"Fair DI = {fair_di:.3f}. IEEE 7003-2024 compliant."
            if fair_di >= 0.9 else
            f"Mitigation applied (DI = {fair_di:.3f}), but still below the 0.9 threshold."
        ),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)