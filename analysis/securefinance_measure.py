
# =====================================================================
# SECUREFINANCE AFRICA
# AICS-201
# TESTING AND EVALUATING AI RISKS
#
# SYSTEM 1: Identity Verification
# SYSTEM 2: Fraud Detection
# SYSTEM 3: Financial Risk Assessment
#
# Synthetic data only. No real customer information.
#
# Random seed: 42
# Output folder: SecureFinance_Results
# =====================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# =====================================================================
# 1. CONFIGURATION
# =====================================================================

SEED = 42
np.random.seed(SEED)

OUTPUT = "AICS201_Fairness_Evidence"
os.makedirs(OUTPUT, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")

IDENTITY_SIZE = 1000
FRAUD_SIZE = 2000
CREDIT_SIZE = 1000

TEST_SIZE = 0.25

print("=" * 75)
print("SECUREFINANCE AFRICA - AI RISK EVALUATION LAB")
print("AICS-201 | MEASURE")
print("=" * 75)

print(f"Random seed: {SEED}")
print(f"Output directory: {OUTPUT}")


# =====================================================================
# 2. HELPER FUNCTIONS
# =====================================================================

def save_csv(df, filename):
    path = os.path.join(OUTPUT, filename)
    df.to_csv(path, index=False)
    print(f"[SAVED] {filename}")


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1 / (1 + np.exp(-x))


def safe_divide(a, b):
    return a / b if b else 0.0


def confusion_metrics(actual, predicted):
    """
    Positive class = 1.

    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    """

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "Recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "F1-Score": f1_score(
            actual, predicted, zero_division=0
        ),
        "False Positive Rate": safe_divide(fp, fp + tn),
        "False Negative Rate": safe_divide(fn, fn + tp),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


def reliability_metrics(original_score, changed_score,
                        original_decision, changed_decision,
                        perturbation):

    correlation = np.corrcoef(
        original_score, changed_score
    )[0, 1]

    if np.isnan(correlation):
        correlation = 0.0

    return {
        "Mean Absolute Prediction Change": np.mean(
            np.abs(original_score - changed_score)
        ),
        "Prediction Correlation": correlation,
        "Decision Change Rate": np.mean(
            original_decision != changed_decision
        ),
        "Perturbation Method": perturbation
    }


def group_identity_metrics(df, group_column):

    records = []

    for group_name, group in df.groupby(group_column):

        actual = group["actual_identity"].to_numpy()
        predicted = group["predicted_verification"].to_numpy()

        genuine = actual == 1
        impostor = actual == 0

        false_acceptances = np.sum(
            impostor & (predicted == 1)
        )

        false_rejections = np.sum(
            genuine & (predicted == 0)
        )

        records.append({
            group_column: group_name,
            "Records": len(group),
            "Accuracy": accuracy_score(actual, predicted),
            "False Acceptance Rate": safe_divide(
                false_acceptances, np.sum(impostor)
            ),
            "False Rejection Rate": safe_divide(
                false_rejections, np.sum(genuine)
            )
        })

    return pd.DataFrame(records)


def group_fraud_metrics(df, group_column):

    records = []

    for group_name, group in df.groupby(group_column):

        metrics = confusion_metrics(
            group["actual_fraud"],
            group["predicted_fraud"]
        )

        records.append({
            group_column: group_name,
            "Transactions": len(group),
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1-Score"],
            "False Positive Rate": metrics["False Positive Rate"],
            "False Negative Rate": metrics["False Negative Rate"]
        })

    return pd.DataFrame(records)


def group_credit_metrics(df, group_column):

    records = []

    for group_name, group in df.groupby(group_column):

        metrics = confusion_metrics(
            group["actual_default"],
            group["predicted_default"]
        )

        records.append({
            group_column: group_name,
            "Applicants": len(group),
            "Approval Rate": group["predicted_decision"].mean(),
            "Accuracy": metrics["Accuracy"],
            "False Positive Rate": metrics["False Positive Rate"],
            "False Negative Rate": metrics["False Negative Rate"],
            "Actual Default Rate": group["actual_default"].mean()
        })

    return pd.DataFrame(records)


# =====================================================================
# 3. SYSTEM 1 - SYNTHETIC IDENTITY VERIFICATION DATA
# =====================================================================

print("\n" + "=" * 75)
print("SYSTEM 1: IDENTITY VERIFICATION")
print("=" * 75)

N = IDENTITY_SIZE

gender = np.random.choice(
    ["Female", "Male"],
    N,
    p=[0.52, 0.48]
)

region = np.random.choice(
    ["Lagos", "Abuja", "Kano", "Rivers", "Other"],
    N,
    p=[0.30, 0.20, 0.17, 0.13, 0.20]
)

network_quality = np.random.choice(
    ["Good", "Average", "Poor"],
    N,
    p=[0.55, 0.30, 0.15]
)

image_quality = np.random.choice(
    ["High", "Medium", "Low"],
    N,
    p=[0.55, 0.30, 0.15]
)

# Ground truth:
# 1 = Legitimate identity
# 0 = Unauthorised / impostor identity

actual_identity = np.random.binomial(
    1, 0.85, N
)

# Synthetic verification score.
# Genuine and impostor attempts have different score distributions.
# Demographic characteristics do not influence these distributions.

base_match_score = np.where(
    actual_identity == 1,
    np.random.normal(86, 12, N),
    np.random.normal(30, 12, N)
)

network_penalty = pd.Series(network_quality).map({
    "Good": 0,
    "Average": 5,
    "Poor": 12
}).to_numpy()

image_penalty = pd.Series(image_quality).map({
    "High": 0,
    "Medium": 5,
    "Low": 12
}).to_numpy()

match_score = np.clip(
    base_match_score - network_penalty - image_penalty,
    0,
    100
)

IDENTITY_THRESHOLD = 65

predicted_verification = (
    match_score >= IDENTITY_THRESHOLD
).astype(int)

identity = pd.DataFrame({
    "customer_id": [
        f"SF-ID-{i:05d}" for i in range(N)
    ],
    "gender": gender,
    "region": region,
    "network_quality": network_quality,
    "image_quality": image_quality,
    "actual_identity": actual_identity,
    "match_score": match_score.round(2),
    "predicted_verification": predicted_verification
})

save_csv(identity, "identity_verification.csv")

# ------------------------------------------------
# IDENTITY METRICS
# ------------------------------------------------

identity_overall = confusion_metrics(
    identity["actual_identity"],
    identity["predicted_verification"]
)

# Actual positive class = legitimate identity (1).
# False acceptance = actual 0, predicted 1.
# False rejection = actual 1, predicted 0.

identity_tn = identity_overall["TN"]
identity_fp = identity_overall["FP"]
identity_fn = identity_overall["FN"]
identity_tp = identity_overall["TP"]

identity_accuracy = identity_overall["Accuracy"]

identity_far = safe_divide(
    identity_fp, identity_fp + identity_tn
)

identity_frr = safe_divide(
    identity_fn, identity_fn + identity_tp
)

print("\nIDENTITY VERIFICATION EVALUATION")

print(f"Overall Accuracy: {identity_accuracy:.4f}")
print(f"False Acceptance Rate: {identity_far:.4f}")
print(f"False Rejection Rate: {identity_frr:.4f}")

identity_summary = pd.DataFrame([{
    "Accuracy": identity_accuracy,
    "False Acceptance Rate": identity_far,
    "False Rejection Rate": identity_frr
}])

save_csv(identity_summary, "identity_metrics.csv")

# ------------------------------------------------
# IDENTITY FAIRNESS BY GENDER
# ------------------------------------------------

identity_gender = group_identity_metrics(
    identity, "gender"
)

save_csv(identity_gender, "identity_fairness.csv")

print("\nIDENTITY FAIRNESS BY GENDER")
print(identity_gender.to_string(index=False))

# ------------------------------------------------
# IDENTITY FAIRNESS BY REGION
# ------------------------------------------------

identity_region = group_identity_metrics(
    identity, "region"
)

save_csv(identity_region, "identity_regional_fairness.csv")

print("\nIDENTITY FAIRNESS BY REGION")
print(identity_region.to_string(index=False))

# ------------------------------------------------
# IDENTITY RELIABILITY
# ------------------------------------------------

# Controlled perturbation:
# Reduce image quality by 5 points and network quality
# by 3 points to represent a slightly worse capture environment.

original_identity_score = identity["match_score"].to_numpy()

changed_identity_score = np.clip(
    original_identity_score - 5 - 3,
    0,
    100
)

original_identity_decision = (
    original_identity_score >= IDENTITY_THRESHOLD
).astype(int)

changed_identity_decision = (
    changed_identity_score >= IDENTITY_THRESHOLD
).astype(int)

identity_reliability = reliability_metrics(
    original_identity_score / 100,
    changed_identity_score / 100,
    original_identity_decision,
    changed_identity_decision,
    "Image quality reduced by 5 points and network quality by 3 points."
)

identity_reliability_df = pd.DataFrame([
    identity_reliability
])

save_csv(
    identity_reliability_df,
    "identity_reliability.csv"
)

print("\nIDENTITY RELIABILITY")
print(identity_reliability_df.to_string(index=False))


# =====================================================================
# 4. SYSTEM 2 - SYNTHETIC FRAUD TRANSACTION DATA
# =====================================================================

print("\n" + "=" * 75)
print("SYSTEM 2: FRAUD DETECTION")
print("=" * 75)

N = FRAUD_SIZE

fraud_gender = np.random.choice(
    ["Female", "Male"],
    N,
    p=[0.52, 0.48]
)

fraud_region = np.random.choice(
    ["Lagos", "Abuja", "Kano", "Rivers", "Other"],
    N
)

transaction_amount = np.random.lognormal(
    mean=7.0,
    sigma=1.0,
    size=N
)

transaction_velocity = np.random.poisson(
    2.5, N
)

device_risk = np.random.uniform(
    0, 1, N
)

location_mismatch = np.random.binomial(
    1, 0.12, N
)

account_age = np.random.randint(
    1, 3000, N
)

merchant_risk = np.random.uniform(
    0, 1, N
)

minutes_since_last = np.random.uniform(
    1, 1440, N
)

# Ground truth generation:
# 1 = Fraudulent transaction
# 0 = Legitimate transaction
#
# The probability depends on transaction risk indicators.
# Gender and region are excluded from the fraud label formula.

fraud_logit = (
    -4.0
    + 0.00012 * transaction_amount
    + 0.24 * transaction_velocity
    + 2.0 * device_risk
    + 1.7 * location_mismatch
    - 0.0004 * account_age
    + 1.4 * merchant_risk
    - 0.001 * minutes_since_last
)

fraud_probability = sigmoid(fraud_logit)

actual_fraud = np.random.binomial(
    1,
    fraud_probability
)

fraud_data = pd.DataFrame({
    "transaction_id": [
        f"SF-TX-{i:06d}" for i in range(N)
    ],
    "gender": fraud_gender,
    "region": fraud_region,
    "transaction_amount": transaction_amount.round(2),
    "transaction_velocity": transaction_velocity,
    "device_risk": device_risk.round(3),
    "location_mismatch": location_mismatch,
    "account_age_days": account_age,
    "merchant_risk": merchant_risk.round(3),
    "minutes_since_last": minutes_since_last.round(2),
    "actual_fraud": actual_fraud
})

# ------------------------------------------------
# TRAIN FRAUD DETECTION MODEL
# ------------------------------------------------

fraud_features = [
    "transaction_amount",
    "transaction_velocity",
    "device_risk",
    "location_mismatch",
    "account_age_days",
    "merchant_risk",
    "minutes_since_last"
]

X_fraud = fraud_data[fraud_features]
y_fraud = fraud_data["actual_fraud"]

Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_fraud,
    y_fraud,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y_fraud
)

fraud_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

fraud_model.fit(Xf_train, yf_train)

FRAUD_THRESHOLD = 0.35

fraud_probabilities = fraud_model.predict_proba(
    Xf_test
)[:, 1]

predicted_fraud = (
    fraud_probabilities >= FRAUD_THRESHOLD
).astype(int)

# Simulated detection time.
# These are synthetic latency estimates, not real measured times.

detection_time = (
    90
    + 8 * Xf_test["transaction_velocity"].to_numpy()
    + 0.002 * Xf_test["transaction_amount"].to_numpy()
    + np.random.normal(0, 15, len(Xf_test))
)

detection_time = np.maximum(
    detection_time, 20
)

fraud_results = fraud_data.loc[Xf_test.index].copy()

fraud_results["predicted_fraud"] = predicted_fraud

fraud_results["fraud_probability"] = (
    fraud_probabilities.round(4)
)

fraud_results["detection_time_ms"] = (
    detection_time.round(2)
)

save_csv(fraud_results, "fraud_transactions.csv")

# ------------------------------------------------
# FRAUD METRICS
# ------------------------------------------------

fraud_metrics = confusion_metrics(
    yf_test,
    predicted_fraud
)

fraud_metrics["ROC-AUC"] = roc_auc_score(
    yf_test, fraud_probabilities
)

fraud_metrics["Average Detection Time (ms)"] = (
    detection_time.mean()
)

fraud_metrics["Median Detection Time (ms)"] = (
    np.median(detection_time)
)

fraud_metrics["95th Percentile Detection Time (ms)"] = (
    np.percentile(detection_time, 95)
)

fraud_metrics_df = pd.DataFrame([
    fraud_metrics
])

save_csv(fraud_metrics_df, "fraud_metrics.csv")

print("\nFRAUD DETECTION EVALUATION")

for name, value in fraud_metrics.items():
    print(f"{name}: {value:.4f}")

# ------------------------------------------------
# FRAUD FAIRNESS BY GENDER
# ------------------------------------------------

fraud_gender_results = group_fraud_metrics(
    fraud_results, "gender"
)

save_csv(
    fraud_gender_results,
    "fraud_fairness.csv"
)

print("\nFRAUD FAIRNESS BY GENDER")
print(fraud_gender_results.to_string(index=False))

# ------------------------------------------------
# FRAUD FAIRNESS BY REGION
# ------------------------------------------------

fraud_region_results = group_fraud_metrics(
    fraud_results, "region"
)

save_csv(
    fraud_region_results,
    "fraud_regional_fairness.csv"
)

print("\nFRAUD FAIRNESS BY REGION")
print(fraud_region_results.to_string(index=False))

# ------------------------------------------------
# FRAUD RELIABILITY
# ------------------------------------------------

# Controlled perturbation:
# Increase transaction amount by 2%.
# Reduce time since the previous transaction by 2 minutes.
#
# This represents a small transaction-value and timing change.

Xf_perturbed = Xf_test.copy()

Xf_perturbed["transaction_amount"] *= 1.02

Xf_perturbed["minutes_since_last"] = np.maximum(
    1,
    Xf_perturbed["minutes_since_last"] - 2
)

original_fraud_probability = fraud_model.predict_proba(
    Xf_test
)[:, 1]

changed_fraud_probability = fraud_model.predict_proba(
    Xf_perturbed
)[:, 1]

original_fraud_decision = (
    original_fraud_probability >= FRAUD_THRESHOLD
).astype(int)

changed_fraud_decision = (
    changed_fraud_probability >= FRAUD_THRESHOLD
).astype(int)

fraud_reliability = reliability_metrics(
    original_fraud_probability,
    changed_fraud_probability,
    original_fraud_decision,
    changed_fraud_decision,
    "Transaction amount increased by 2%; time since last transaction reduced by 2 minutes."
)

fraud_reliability_df = pd.DataFrame([
    fraud_reliability
])

save_csv(
    fraud_reliability_df,
    "fraud_reliability.csv"
)

print("\nFRAUD RELIABILITY")
print(fraud_reliability_df.to_string(index=False))


# =====================================================================
# 5. SYSTEM 3 - SYNTHETIC CREDIT RISK DATA
# =====================================================================

print("\n" + "=" * 75)
print("SYSTEM 3: FINANCIAL RISK ASSESSMENT")
print("=" * 75)

N = CREDIT_SIZE

credit_gender = np.random.choice(
    ["Female", "Male"],
    N,
    p=[0.52, 0.48]
)

credit_region = np.random.choice(
    ["Lagos", "Abuja", "Kano", "Rivers", "Other"],
    N
)

age = np.random.randint(
    21, 66, N
)

income = np.random.lognormal(
    mean=12.0,
    sigma=0.65,
    size=N
)

debt_ratio = np.random.beta(
    2, 5, N
)

credit_history = np.random.randint(
    0, 16, N
)

employment_years = np.random.randint(
    0, 25, N
)

loan_amount = np.random.lognormal(
    mean=12.0,
    sigma=0.55,
    size=N
)

# Ground truth:
# 1 = Customer defaults.
# 0 = Customer repays.
#
# Demographic attributes are excluded from the formula.

default_logit = (
    -2.5
    + 3.0 * debt_ratio
    + 0.000015 * loan_amount
    - 0.000009 * income
    - 0.10 * credit_history
    - 0.04 * employment_years
)

default_probability = sigmoid(default_logit)

actual_default = np.random.binomial(
    1, default_probability
)

credit_data = pd.DataFrame({
    "applicant_id": [
        f"SF-LOAN-{i:05d}" for i in range(N)
    ],
    "gender": credit_gender,
    "region": credit_region,
    "age": age,
    "income": income.round(2),
    "loan_amount": loan_amount.round(2),
    "debt_ratio": debt_ratio.round(3),
    "credit_history_years": credit_history,
    "employment_years": employment_years,
    "actual_default": actual_default
})

save_csv(
    credit_data,
    "credit_applicants.csv"
)

# ------------------------------------------------
# TRAIN CREDIT RISK MODEL
# ------------------------------------------------

credit_features = [
    "age",
    "income",
    "loan_amount",
    "debt_ratio",
    "credit_history_years",
    "employment_years"
]

X_credit = credit_data[credit_features]

y_credit = credit_data["actual_default"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_credit,
    y_credit,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y_credit
)

credit_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

credit_model.fit(Xc_train, yc_train)

credit_risk_probability = credit_model.predict_proba(
    Xc_test
)[:, 1]

CREDIT_THRESHOLD = 0.30

# Positive class for default prediction = default (1).
# Predicted default = 1.
# Predicted no default = 0.

predicted_default = (
    credit_risk_probability >= CREDIT_THRESHOLD
).astype(int)

# Lending decision:
# Approve loan when predicted default risk is below 30%.

predicted_decision = (
    credit_risk_probability < CREDIT_THRESHOLD
).astype(int)

credit_results = credit_data.loc[Xc_test.index].copy()

credit_results["predicted_default_probability"] = (
    credit_risk_probability.round(4)
)

credit_results["predicted_default"] = predicted_default

credit_results["predicted_decision"] = predicted_decision

save_csv(
    credit_results,
    "credit_risk_results.csv"
)

# ------------------------------------------------
# CREDIT MODEL PERFORMANCE
# ------------------------------------------------

credit_metrics = confusion_metrics(
    yc_test,
    predicted_default
)

credit_metrics["ROC-AUC"] = roc_auc_score(
    yc_test,
    credit_risk_probability
)

credit_metrics["Approval Rate"] = (
    predicted_decision.mean()
)

credit_metrics_df = pd.DataFrame([
    credit_metrics
])

save_csv(
    credit_metrics_df,
    "credit_metrics.csv"
)

print("\nCREDIT RISK MODEL EVALUATION")

for name, value in credit_metrics.items():
    print(f"{name}: {value:.4f}")

print("\nPOSITIVE CLASS DEFINITION")
print("Default = 1")
print("No default = 0")
print("False positive = Predicted default, actual repayment.")
print("False negative = Predicted repayment, actual default.")

# ------------------------------------------------
# CREDIT FAIRNESS BY GENDER
# ------------------------------------------------

credit_gender_results = group_credit_metrics(
    credit_results,
    "gender"
)

save_csv(
    credit_gender_results,
    "credit_gender_fairness.csv"
)

print("\nCREDIT FAIRNESS BY GENDER")
print(credit_gender_results.to_string(index=False))

# ------------------------------------------------
# CREDIT FAIRNESS BY REGION
# ------------------------------------------------

credit_region_results = group_credit_metrics(
    credit_results,
    "region"
)

save_csv(
    credit_region_results,
    "credit_region_fairness.csv"
)

print("\nCREDIT FAIRNESS BY REGION")
print(credit_region_results.to_string(index=False))

# ------------------------------------------------
# CREDIT APPROVAL RATE DIFFERENCES
# ------------------------------------------------

gender_approval = credit_gender_results.set_index(
    "gender"
)["Approval Rate"]

if "Female" in gender_approval and "Male" in gender_approval:

    gender_approval_difference = abs(
        gender_approval["Female"] -
        gender_approval["Male"]
    )

else:
    gender_approval_difference = np.nan

region_approval_range = (
    credit_region_results["Approval Rate"].max()
    - credit_region_results["Approval Rate"].min()
)

gender_fpr_difference = (
    credit_gender_results["False Positive Rate"].max()
    - credit_gender_results["False Positive Rate"].min()
)

gender_fnr_difference = (
    credit_gender_results["False Negative Rate"].max()
    - credit_gender_results["False Negative Rate"].min()
)

region_fpr_difference = (
    credit_region_results["False Positive Rate"].max()
    - credit_region_results["False Positive Rate"].min()
)

region_fnr_difference = (
    credit_region_results["False Negative Rate"].max()
    - credit_region_results["False Negative Rate"].min()
)

approval_differences = pd.DataFrame([{
    "Female_vs_Male_Approval_Difference":
        gender_approval_difference,
    "Regional_Approval_Rate_Range":
        region_approval_range,
    "Gender_FPR_Difference":
        gender_fpr_difference,
    "Gender_FNR_Difference":
        gender_fnr_difference,
    "Regional_FPR_Difference":
        region_fpr_difference,
    "Regional_FNR_Difference":
        region_fnr_difference
}])

save_csv(
    approval_differences,
    "credit_approval_differences.csv"
)

print("\nCREDIT APPROVAL RATE DIFFERENCES")
print(approval_differences.to_string(index=False))


# =====================================================================
# 6. CREDIT MODEL EXPLAINABILITY
# =====================================================================

print("\n" + "=" * 75)
print("CREDIT MODEL EXPLAINABILITY")
print("=" * 75)

# Method: Permutation Importance.
#
# Each feature is shuffled repeatedly.
# The reduction in ROC-AUC measures its predictive importance.
#
# This is not evidence of causation.

importance_result = permutation_importance(
    credit_model,
    Xc_test,
    yc_test,
    scoring="roc_auc",
    n_repeats=15,
    random_state=SEED
)

importance = pd.DataFrame({
    "Feature": credit_features,
    "Importance": importance_result.importances_mean,
    "Standard Deviation": importance_result.importances_std
})

# Convert positive importance values into percentages.
# If all mean importance values are non-positive,
# absolute magnitudes are used for a descriptive comparison.

importance["Contribution Basis"] = np.maximum(
    importance["Importance"], 0
)

if importance["Contribution Basis"].sum() == 0:
    importance["Contribution Basis"] = (
        importance["Importance"].abs()
    )

total_importance = importance["Contribution Basis"].sum()

if total_importance > 0:
    importance["Contribution Percentage"] = (
        importance["Contribution Basis"]
        / total_importance * 100
    )
else:
    importance["Contribution Percentage"] = 0

importance = importance.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)

save_csv(
    importance,
    "credit_explainability.csv"
)

print("\nCREDIT MODEL EXPLAINABILITY")

for _, row in importance.iterrows():

    print(
        f"{row['Feature']}: "
        f"{row['Contribution Percentage']:.2f}%"
    )

most_influential = importance.iloc[0]["Feature"]

least_influential = importance.iloc[-1]["Feature"]

print("\nMost Influential Feature:", most_influential)
print("Least Influential Feature:", least_influential)

print("Method: Permutation Importance using ROC-AUC.")
print("Interpretation: Predictive importance, not causation.")


# =====================================================================
# 7. CREDIT RELIABILITY AND STABILITY
# =====================================================================

# Controlled perturbation:
# Income decreases by 2%.
# Loan amount increases by 2%.
# Debt ratio increases by 0.01 (one percentage point).
#
# These represent small hypothetical changes in financial inputs.

Xc_perturbed = Xc_test.copy()

Xc_perturbed["income"] *= 0.98

Xc_perturbed["loan_amount"] *= 1.02

Xc_perturbed["debt_ratio"] = np.clip(
    Xc_perturbed["debt_ratio"] + 0.01,
    0,
    1
)

original_credit_probability = credit_model.predict_proba(
    Xc_test
)[:, 1]

changed_credit_probability = credit_model.predict_proba(
    Xc_perturbed
)[:, 1]

original_credit_decision = (
    original_credit_probability >= CREDIT_THRESHOLD
).astype(int)

changed_credit_decision = (
    changed_credit_probability >= CREDIT_THRESHOLD
).astype(int)

credit_reliability = reliability_metrics(
    original_credit_probability,
    changed_credit_probability,
    original_credit_decision,
    changed_credit_decision,
    "Income decreased by 2%, loan amount increased by 2%, debt ratio increased by 0.01."
)

credit_reliability_df = pd.DataFrame([
    credit_reliability
])

save_csv(
    credit_reliability_df,
    "credit_reliability.csv"
)

print("\nCREDIT RELIABILITY")
print(credit_reliability_df.to_string(index=False))


# =====================================================================
# 8. VISUALIZATION DASHBOARD
# =====================================================================

print("\n" + "=" * 75)
print("GENERATING AI EVALUATION DASHBOARD")
print("=" * 75)

fig, axes = plt.subplots(
    2, 3,
    figsize=(19, 12)
)

fig.suptitle(
    "SECUREFINANCE AFRICA | AI RISK EVALUATION",
    fontsize=18,
    fontweight="bold"
)

# 1. Identity verification outcomes.

sns.countplot(
    data=identity,
    x="predicted_verification",
    hue="predicted_verification",
    legend=False,
    ax=axes[0, 0]
)

axes[0, 0].set_title("Identity Verification Results")
axes[0, 0].set_xticks([0, 1], ["Rejected", "Accepted"])

# 2. Identity false rejection by gender.

sns.barplot(
    data=identity_gender,
    x="gender",
    y="False Rejection Rate",
    hue="gender",
    legend=False,
    ax=axes[0, 1]
)

axes[0, 1].set_title("Identity False Rejection by Gender")

# 3. Fraud detection scores.

sns.histplot(
    data=fraud_results,
    x="fraud_probability",
    hue="actual_fraud",
    bins=25,
    common_norm=False,
    ax=axes[0, 2]
)

axes[0, 2].set_title("Fraud Probability Distribution")

# 4. Fraud precision and recall by region.

fraud_region_plot = fraud_region_results.melt(
    id_vars="region",
    value_vars=["Precision", "Recall"],
    var_name="Metric",
    value_name="Score"
)

sns.barplot(
    data=fraud_region_plot,
    x="region",
    y="Score",
    hue="Metric",
    ax=axes[1, 0]
)

axes[1, 0].set_title("Fraud Detection by Region")
axes[1, 0].tick_params(axis="x", rotation=25)

# 5. Credit approval by gender.

sns.barplot(
    data=credit_gender_results,
    x="gender",
    y="Approval Rate",
    hue="gender",
    legend=False,
    ax=axes[1, 1]
)

axes[1, 1].set_title("Credit Approval by Gender")
axes[1, 1].set_ylim(0, 1)

# 6. Credit explainability.

sns.barplot(
    data=importance,
    x="Contribution Percentage",
    y="Feature",
    ax=axes[1, 2]
)

axes[1, 2].set_title("Credit Feature Importance (%)")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT, "stage3_4_dashboard.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================================================
# 9. FINAL SUMMARY REPORT
# =====================================================================

print("\n" + "=" * 75)
print("AI EVALUATION - FINAL SUMMARY")
print("=" * 75)

summary = []

summary.append("SECUREFINANCE AFRICA")
summary.append("AICS-201 - MEASURE")
summary.append("AI EVALUATION - FINAL SUMMARY")
summary.append("=" * 70)

# ------------------------------------------------
# A. IDENTITY VERIFICATION
# ------------------------------------------------

summary.append("\nA. IDENTITY VERIFICATION")

summary.append(
    f"Overall Accuracy: {identity_accuracy:.4f}"
)

summary.append(
    f"False Acceptance Rate: {identity_far:.4f}"
)

summary.append(
    f"False Rejection Rate: {identity_frr:.4f}"
)

summary.append("\nIdentity Fairness by Gender:")
summary.append(identity_gender.to_string(index=False))

summary.append("\nIdentity Fairness by Region:")
summary.append(identity_region.to_string(index=False))

summary.append("\nIdentity Reliability:")
summary.append(identity_reliability_df.to_string(index=False))


# ------------------------------------------------
# B. FRAUD DETECTION
# ------------------------------------------------

summary.append("\n\nB. FRAUD DETECTION")

for metric in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "False Positive Rate",
    "False Negative Rate",
    "Average Detection Time (ms)",
    "Median Detection Time (ms)",
    "95th Percentile Detection Time (ms)"
]:
    summary.append(
        f"{metric}: {fraud_metrics[metric]:.4f}"
    )

summary.append("\nFraud Fairness by Gender:")
summary.append(fraud_gender_results.to_string(index=False))

summary.append("\nFraud Fairness by Region:")
summary.append(fraud_region_results.to_string(index=False))

summary.append("\nFraud Reliability:")
summary.append(fraud_reliability_df.to_string(index=False))


# ------------------------------------------------
# C. CREDIT RISK ASSESSMENT
# ------------------------------------------------

summary.append("\n\nC. CREDIT RISK ASSESSMENT")

summary.append("\nGender Fairness:")
summary.append(credit_gender_results.to_string(index=False))

summary.append("\nRegional Fairness:")
summary.append(credit_region_results.to_string(index=False))

summary.append("\nApproval Rate Differences:")
summary.append(approval_differences.to_string(index=False))

summary.append("\nFeature Importance:")
summary.append(importance.to_string(index=False))

summary.append(
    f"\nMost Influential Feature: {most_influential}"
)

summary.append(
    f"Least Influential Feature: {least_influential}"
)

summary.append("\nCredit Reliability:")
summary.append(credit_reliability_df.to_string(index=False))


# ------------------------------------------------
# D. TESTING INFORMATION
# ------------------------------------------------

summary.append("\n\nD. TESTING INFORMATION")

summary.append(f"Random Seed: {SEED}")

summary.append(f"Identity Dataset Size: {IDENTITY_SIZE}")

summary.append(f"Fraud Dataset Size: {FRAUD_SIZE}")

summary.append(f"Credit Dataset Size: {CREDIT_SIZE}")

summary.append(f"Training/Test Split: 75% / 25%")

summary.append("Identity Method: Simulated matching score and threshold.")

summary.append("Fraud Model: StandardScaler + Logistic Regression.")

summary.append("Credit Model: StandardScaler + Logistic Regression.")

summary.append("Explainability: Permutation Importance using ROC-AUC.")

summary.append("Identity Positive Class: Legitimate identity = 1.")

summary.append("Fraud Positive Class: Fraudulent transaction = 1.")

summary.append("Credit Positive Class: Default = 1.")

summary.append("Identity threshold: 65/100.")

summary.append("Fraud threshold: 0.35.")

summary.append("Credit default threshold: 0.30.")

summary.append(
    "Reliability: Controlled input perturbations and "
    "prediction stability comparisons."
)


# ------------------------------------------------
# E. LIMITATIONS
# ------------------------------------------------

summary.append("\n\nE. LIMITATIONS")

summary.append(
    "1. All datasets and labels are synthetically generated."
)

summary.append(
    "2. Identity verification is a simulated score-based system, "
    "not a trained biometric recognition model."
)

summary.append(
    "3. Fraud detection uses a trained model on synthetic "
    "transaction data."
)

summary.append(
    "4. Credit assessment uses a trained model on synthetic "
    "applicant data."
)

summary.append(
    "5. Detection times are simulated, not measured "
    "production response times."
)

summary.append(
    "6. Fairness differences are diagnostic indicators and "
    "do not automatically establish discrimination."
)

summary.append(
    "7. Permutation importance does not establish causation."
)

summary.append(
    "8. Controlled perturbation tests do not prove "
    "real-world reliability."
)

summary.append(
    "9. Synthetic performance cannot establish production "
    "accuracy or legal compliance."
)


# ------------------------------------------------
# F. GENERATED EVIDENCE
# ------------------------------------------------

summary.append("\n\nF. GENERATED EVIDENCE")

for filename in sorted(os.listdir(OUTPUT)):
    summary.append(f"- {filename}")

summary.append("\n" + "=" * 70)
summary.append("END OF EVALUATION")
summary.append("=" * 70)

summary_text = "\n".join(summary)

report_path = os.path.join(
    OUTPUT,
    "final_summary_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:
    file.write(summary_text)

print(summary_text)


# =====================================================================
# 10. FINAL EXECUTION STATUS
# =====================================================================

print("\n" + "=" * 75)
print("STAGE MEASURE COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nDATASET SIZES")
print("Identity verification:", len(identity))
print("Fraud transactions:", len(fraud_data))
print("Credit applicants:", len(credit_data))

print("\nGENERATED FILES")

for filename in sorted(os.listdir(OUTPUT)):
    print("[OK]", filename)

print("\nAll evidence saved in:", OUTPUT)

print("=" * 75)
