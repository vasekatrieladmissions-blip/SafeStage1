"""
SafeStage ML Functions
======================
Four self-contained analytical modules:
  1. Classification  – Random Forest → predict User_Type (Fan / Scalper / Bot)
  2. Clustering      – K-Means      → segment users into Personas
  3. Regression      – Linear Reg   → predict Max_Price_Willing_to_Pay_USD
  4. Association     – Apriori      → market-basket rules on features / challenges

Each function accepts a cleaned DataFrame and returns a results dict that the
Streamlit app can render directly.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLASSIFICATION — Random Forest → predict User_Type
# ─────────────────────────────────────────────────────────────────────────────

def run_classification(df: pd.DataFrame) -> dict:
    """
    Predict User_Type (Fan / Scalper / Bot) from behavioural signals.

    Features used
    -------------
    - Time_to_Click_Buy_ms        : reaction speed (bots are near-instant)
    - Mouse_Movement_Pattern_Code : 0=Linear, 1=Erratic, 2=Natural
    - Session_Count_Last_24h      : page-refresh frequency
    - Liveness_Check_Duration_sec : biometric check speed
    - Verified_Status_Code        : identity verified?
    - Account_Age_Days            : account maturity
    - Historical_Spend_USD        : cumulative spend

    Target
    ------
    User_Type_Code : 0=Bot, 1=Scalper, 2=Fan
    """
    FEATURES = [
        "Time_to_Click_Buy_ms",
        "Mouse_Movement_Pattern_Code",
        "Session_Count_Last_24h",
        "Liveness_Check_Duration_sec",
        "Verified_Status_Code",
        "Account_Age_Days",
        "Historical_Spend_USD",
    ]
    TARGET = "User_Type_Code"
    LABEL_MAP = {0: "Bot", 1: "Scalper", 2: "Fan"}

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred)

    # Feature importance
    fi = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)

    return {
        "model": clf,
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importance": fi,
        "label_map": LABEL_MAP,
        "features": FEATURES,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def predict_user_type(clf, features: list, label_map: dict, input_values: dict) -> dict:
    """
    Single-row inference for the Streamlit interactive sliders.

    Parameters
    ----------
    clf          : trained RandomForestClassifier
    features     : ordered list of feature names
    label_map    : {code: label}
    input_values : {feature_name: value}

    Returns
    -------
    dict with predicted label, code, and class probabilities
    """
    row = pd.DataFrame([input_values])[features]
    code = clf.predict(row)[0]
    proba = clf.predict_proba(row)[0]
    return {
        "predicted_label": label_map[code],
        "predicted_code": int(code),
        "probabilities": {label_map[i]: round(float(p), 4) for i, p in enumerate(proba)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLUSTERING — K-Means → User Personas
# ─────────────────────────────────────────────────────────────────────────────

# Persona labels are assigned by inspecting cluster centroids:
#   High spend + many sessions + young  → "The Superfan"
#   Low spend  + few sessions           → "The Casual"
#   Very high sessions + low liveness   → "The Bot / Scalper"
PERSONA_LABELS = {
    0: "🎸 The Superfan",
    1: "🎟️ The Casual",
    2: "🤖 The Bot/Scalper",
}

def run_clustering(df: pd.DataFrame, n_clusters: int = 3) -> dict:
    """
    Segment users into behavioural personas using K-Means.

    Features used
    -------------
    - Age
    - Historical_Spend_USD
    - Session_Count_Last_24h
    - Liveness_Check_Duration_sec
    - Account_Age_Days

    Returns cluster assignments, centroids, inertia, and labelled personas.
    """
    FEATURES = [
        "Age",
        "Historical_Spend_USD",
        "Session_Count_Last_24h",
        "Liveness_Check_Duration_sec",
        "Account_Age_Days",
    ]

    X = df[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, random_state=42)
    labels = km.fit_predict(X_scaled)

    df_result = df.copy()
    df_result["Cluster"] = labels

    # Map clusters → human-readable persona by centroid spend (descending)
    centroid_spend = (
        df_result.groupby("Cluster")["Historical_Spend_USD"].mean().sort_values(ascending=False)
    )
    persona_keys = list(PERSONA_LABELS.keys())
    cluster_to_persona = {
        int(cluster): PERSONA_LABELS[persona_keys[rank]]
        for rank, cluster in enumerate(centroid_spend.index)
    }
    df_result["Persona"] = df_result["Cluster"].map(cluster_to_persona)

    # Cluster summary statistics
    summary = (
        df_result.groupby("Persona")[FEATURES]
        .mean()
        .round(1)
        .reset_index()
    )

    # Inertia curve data (elbow method, k=2..8)
    inertias = {}
    for k in range(2, 9):
        inertias[k] = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled).inertia_

    return {
        "model": km,
        "scaler": scaler,
        "labels": labels,
        "df_clustered": df_result,
        "cluster_to_persona": cluster_to_persona,
        "centroids_original": scaler.inverse_transform(km.cluster_centers_),
        "centroid_features": FEATURES,
        "summary": summary,
        "inertias": inertias,
        "features": FEATURES,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. REGRESSION — Linear Regression → predict Max_Price_Willing_to_Pay_USD
# ─────────────────────────────────────────────────────────────────────────────

def run_regression(df: pd.DataFrame) -> dict:
    """
    Predict a user's maximum willingness-to-pay for a high-demand event.

    Features used
    -------------
    - Historical_Spend_USD  : total past spend (strongest predictor)
    - Age                   : older fans often spend more
    - Account_Age_Days      : platform loyalty proxy
    - Session_Count_Last_24h: engagement intensity

    Target
    ------
    Max_Price_Willing_to_Pay_USD
    """
    FEATURES = [
        "Historical_Spend_USD",
        "Age",
        "Account_Age_Days",
        "Session_Count_Last_24h",
    ]
    TARGET = "Max_Price_Willing_to_Pay_USD"

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    coeff_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": reg.coef_,
        "Abs_Coefficient": np.abs(reg.coef_),
    }).sort_values("Abs_Coefficient", ascending=False)

    return {
        "model": reg,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "intercept": reg.intercept_,
        "coefficients": coeff_df,
        "features": FEATURES,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def predict_max_price(reg, features: list, input_values: dict) -> float:
    """
    Single-row inference for the Streamlit regression slider panel.
    Returns the predicted max price in USD.
    """
    row = pd.DataFrame([input_values])[features]
    return float(reg.predict(row)[0])


# ─────────────────────────────────────────────────────────────────────────────
# 4. ASSOCIATION RULES — Apriori on features / challenges
# ─────────────────────────────────────────────────────────────────────────────

def run_association_rules(
    df: pd.DataFrame,
    min_support: float = 0.10,
    min_confidence: float = 0.50,
    min_lift: float = 1.0,
) -> dict:
    """
    Find co-occurrence patterns between:
      - Primary_Interest   (Sports / Music / Theatre)
      - Preferred_Feature  (Face-value Resale / Group Booking / Early Access)
      - Challenges_Faced   (High Prices / Bot Competition / Interface Complexity)

    Example insight:
      {Bot Competition} → {Face-value Resale}
      (users who face bots are most likely to want a resale-proof platform)

    Returns rules DataFrame plus the most actionable rule highlights.
    """
    # Build one-hot encoded transaction matrix
    item_cols = ["Primary_Interest", "Preferred_Feature", "Challenges_Faced"]
    transactions = df[item_cols].astype(str).values.tolist()

    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    basket_df = pd.DataFrame(te_array, columns=te.columns_)

    freq_items = apriori(basket_df, min_support=min_support, use_colnames=True)

    if freq_items.empty:
        return {"rules": pd.DataFrame(), "frequent_itemsets": freq_items, "error": "No frequent itemsets found — try lowering min_support."}

    rules = association_rules(freq_items, metric="lift", min_threshold=min_lift)
    rules = rules[rules["confidence"] >= min_confidence].sort_values("lift", ascending=False)

    # Tag rules with antecedent/consequent as readable strings
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

    # Highlight the top insight: Bot Competition → Face-value Resale
    highlight = rules[
        rules["antecedents_str"].str.contains("Bot Competition") &
        rules["consequents_str"].str.contains("Face-value Resale")
    ]

    return {
        "rules": rules,
        "frequent_itemsets": freq_items,
        "highlight_rules": highlight,
        "basket_df": basket_df,
        "params": {
            "min_support": min_support,
            "min_confidence": min_confidence,
            "min_lift": min_lift,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv("safestage_cleaned.csv")

    print("=" * 60)
    print("1. CLASSIFICATION")
    clf_results = run_classification(df)
    print(f"   Accuracy : {clf_results['accuracy']:.4f}")
    print(f"   Top feature: {clf_results['feature_importance'].idxmax()}")

    print("\n2. CLUSTERING")
    clus_results = run_clustering(df)
    print(f"   Personas : {list(clus_results['cluster_to_persona'].values())}")

    print("\n3. REGRESSION")
    reg_results = run_regression(df)
    print(f"   R²   : {reg_results['r2']:.4f}")
    print(f"   MAE  : ${reg_results['mae']:.2f}")
    print(f"   RMSE : ${reg_results['rmse']:.2f}")

    print("\n4. ASSOCIATION RULES")
    ar_results = run_association_rules(df)
    print(f"   Total rules found : {len(ar_results['rules'])}")
    if not ar_results["highlight_rules"].empty:
        h = ar_results["highlight_rules"].iloc[0]
        print(f"   Key rule : {{{h['antecedents_str']}}} → {{{h['consequents_str']}}}")
        print(f"   Confidence={h['confidence']:.2f}  Lift={h['lift']:.2f}")

    print("\nAll functions OK ✓")
