# 🎫 SafeStage Analytics Platform

> **Making live events scalper-proof through identity-linked ticketing and machine learning.**

---

## The Problem

Ticket scalping costs genuine fans billions annually. Within seconds of a sale going live, automated bots and professional scalpers bulk-purchase inventory and relist it at 300–1000% mark-ups. Traditional ticketing platforms have no reliable way to distinguish a genuine fan from a bot or a reseller at the point of purchase.

**SafeStage** solves this at the identity layer — not the payment layer.

---

## The Solution

SafeStage uses **biometric identity verification** combined with **machine learning behavioural analysis** to create a closed-loop ticketing ecosystem:

| Layer | Mechanism | ML Application |
|---|---|---|
| **Identity** | Government ID + liveness check | Verified_Status, Liveness_Check_Duration |
| **Behaviour** | Mouse movement + click timing | Random Forest Classification |
| **Resale** | Face-value only closed marketplace | Regression-based price ceiling |
| **Personas** | Audience segmentation | K-Means Clustering |
| **Insights** | Co-occurrence patterns | Apriori Association Rules |

---

## Dashboard Pages

### 📊 Overview & EDA
Exploratory analysis of the 2,000-user synthetic dataset. Includes a correlation heatmap and the key bot-detection histogram showing that bots click 'Buy' in under 500ms — a physically impossible speed for humans.

### 🤖 Scalper Detection (ML)
- **Random Forest Classifier** predicts whether a user is a Fan, Scalper, or Bot from 7 behavioural signals.
- **Interactive live classifier** — adjust sliders to simulate a user and get a real-time prediction.
- **Linear Regression model** predicts each user's maximum willingness to pay, enabling a personalised face-value ceiling at checkout.

### 👥 User Personas
K-Means clustering segments users into three archetypes:
- 🎸 **The Superfan** — high spend, high engagement, genuine fan
- 🎟️ **The Casual** — moderate spend, infrequent attendance
- 🤖 **The Bot/Scalper** — extreme session counts, near-instant click times

Includes a 3D scatter plot and elbow curve for cluster validation.

### 📈 Market Insights
Apriori association rules reveal:
- Users citing **"Bot Competition"** as a challenge are the most likely to want **"Face-value Resale"** — directly validating SafeStage's core proposition.
- Adjustable support/confidence/lift parameters for live exploration.

---

## Project Structure

```
safestage/
│
├── app.py                    # Streamlit dashboard (main entry point)
├── ml_functions.py           # All 4 ML modules (Classification, Clustering, Regression, Association)
├── safestage_cleaned.csv     # Cleaned 2,000-row synthetic dataset
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Quickstart

### 1. Clone / unzip the project
```bash
unzip safestage.zip
cd safestage
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the dashboard
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` in your browser.

---

## Data Science Techniques Used

| Technique | Library | Purpose |
|---|---|---|
| Random Forest | `scikit-learn` | Classify users as Fan / Scalper / Bot |
| K-Means | `scikit-learn` | Segment users into behavioural personas |
| Linear Regression | `scikit-learn` | Predict maximum willingness to pay |
| Apriori | `mlxtend` | Discover feature co-occurrence patterns |
| EDA / Visualisation | `plotly`, `seaborn` | Correlation heatmaps, distribution plots |

---

## Dataset

The `safestage_cleaned.csv` is a **synthetic** 2,000-row dataset generated to represent realistic distributions across three user archetypes. It contains 29 features including behavioural metrics, identity verification signals, financial history, and preference data.

No real user data is used.

---

## Deployment (Streamlit Community Cloud)

1. Push the `safestage/` folder to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Point it to your repo, set the main file to `app.py`.
4. Click **Deploy** — it will auto-install from `requirements.txt`.

---

## License

MIT — free to use, modify, and deploy.

---

*Built as a data science portfolio project demonstrating end-to-end ML application development for the live events industry.*
