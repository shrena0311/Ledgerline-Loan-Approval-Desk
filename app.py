import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Ledgerline — Loan Approval Desk",
    page_icon="🖋️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CAT_COLS = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]

# --------------------------------------------------------------------------
# Style: a bank ledger sitting on a clerk's desk, not a dashboard.
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Style: a bank ledger sitting on a clerk's desk, not a dashboard.
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Style: a bank ledger sitting on a clerk's desk, not a dashboard.
# --------------------------------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,500&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">

    <style>

    :root{
      --ink:#101826;
      --ink-2:#182236;
      --paper:#F1E8D6;
      --paper-2:#e9dcbd;
      --rule:#cdbd94;
      --brass:#B7863C;
      --brass-2:#8f6a2e;
      --sage:#3f6b53;
      --rust:#9a4a35;
      --text-on-paper:#2a2216;
      --text-dim:#8b93a3;
      --text-light:#e7e2d3;
    }

    /* ---------- Global page ---------- */

    html, body, [class*="css"]{
      font-family:'Inter', sans-serif;
    }

    .stApp{
      background:
        repeating-linear-gradient(
          180deg,
          rgba(255,255,255,0.018) 0px,
          rgba(255,255,255,0.018) 1px,
          transparent 1px,
          transparent 34px
        ),
        radial-gradient(
          ellipse at 20% -10%,
          #1b2740 0%,
          var(--ink) 55%
        );
      color:var(--text-light);
    }

    #MainMenu,
    footer,
    header[data-testid="stHeader"]{
      visibility:hidden;
      height:0;
    }

    .block-container{
      padding-top:2.6rem;
      max-width:1180px;
    }


    /* ---------- Header / letterhead ---------- */

    .eyebrow-row{
      display:flex;
      justify-content:space-between;
      align-items:baseline;
      border-bottom:1px solid rgba(231,226,211,0.18);
      padding-bottom:14px;
      margin-bottom:26px;
      font-family:'IBM Plex Mono', monospace;
      letter-spacing:0.12em;
      font-size:0.72rem;
      color:var(--text-dim);
      text-transform:uppercase;
    }

    .eyebrow-row span.mark{
      color:var(--brass);
    }

    .masthead h1{
      font-family:'Fraunces', serif;
      font-weight:600;
      font-size:2.7rem;
      line-height:1.08;
      color:var(--text-light);
      margin-bottom:6px;
      letter-spacing:-0.01em;
    }

    .masthead h1 em{
      font-style:italic;
      color:var(--brass);
      font-weight:500;
    }

    .masthead p.sub{
      font-family:'Inter', sans-serif;
      color:var(--text-dim);
      font-size:1rem;
      max-width:640px;
      margin-bottom:30px;
    }


    /* ---------- Ledger card ---------- */

    .ledger-card{
      background:var(--paper);
      border-radius:3px;
      padding:28px 30px 22px 30px;
      box-shadow:
        0 18px 40px -14px rgba(0,0,0,0.55),
        0 2px 0 rgba(0,0,0,0.08);
      position:relative;
      color:var(--text-on-paper);
    }

    .ledger-card::before{
      content:"";
      position:absolute;
      inset:10px;
      border:1px solid rgba(42,34,22,0.14);
      pointer-events:none;
    }

    .card-label{
      font-family:'IBM Plex Mono', monospace;
      font-size:0.68rem;
      letter-spacing:0.14em;
      text-transform:uppercase;
      color:var(--brass-2);
      margin-bottom:4px;
    }

    .card-title{
      font-family:'Fraunces', serif;
      font-weight:600;
      font-size:1.35rem;
      margin-bottom:14px;
      border-bottom:1px dashed var(--rule);
      padding-bottom:12px;
      color:var(--text-on-paper) !important;
    }


    /* ---------- Streamlit form ---------- */

    div[data-testid="stForm"]{
      background:var(--paper);
      border-radius:3px;
      padding:26px 30px 10px 30px;
      box-shadow:
        0 18px 40px -14px rgba(0,0,0,0.55),
        0 2px 0 rgba(0,0,0,0.08);
      color:var(--text-on-paper);
    }


    /* ---------- Form labels ---------- */

    div[data-testid="stForm"] label p,
    div[data-testid="stForm"] label{
      font-family:'IBM Plex Mono', monospace !important;
      font-size:0.72rem !important;
      letter-spacing:0.05em;
      text-transform:uppercase;
      color:var(--text-on-paper) !important;
      opacity:0.72;
    }

    div[data-testid="stForm"] .stRadio label p,
    div[data-testid="stForm"] .stSelectbox label p,
    div[data-testid="stForm"] .stNumberInput label p{
      color:var(--text-on-paper) !important;
      opacity:0.72;
    }


    /* ==========================================================
       INPUTS / SELECTBOXES
       The important fix: light text inside the dark controls.
       ========================================================== */

    /* Text and number inputs */

    div[data-testid="stForm"] input,
    div[data-testid="stForm"] input[type="text"],
    div[data-testid="stForm"] input[type="number"]{
      background:#252630 !important;
      color:#F1E8D6 !important;
      -webkit-text-fill-color:#F1E8D6 !important;

      border:none !important;
      border-bottom:1.5px solid var(--rule) !important;
      border-radius:0 !important;

      font-family:'IBM Plex Mono', monospace !important;
      font-size:0.92rem !important;

      box-shadow:none !important;
      caret-color:var(--brass) !important;
    }

    /* Number input value */

    div[data-testid="stForm"] input[type="number"]{
      color:#F1E8D6 !important;
      -webkit-text-fill-color:#F1E8D6 !important;
    }

    /* Selectbox itself */

    div[data-testid="stForm"] div[data-baseweb="select"] > div{
      background:#252630 !important;
      color:#F1E8D6 !important;

      border:none !important;
      border-bottom:1.5px solid var(--rule) !important;
      border-radius:0 !important;

      box-shadow:none !important;
    }

    /* Selected value inside selectbox */

    div[data-testid="stForm"] div[data-baseweb="select"] span,
    div[data-testid="stForm"] div[data-baseweb="select"] input{
      color:#F1E8D6 !important;
      -webkit-text-fill-color:#F1E8D6 !important;
      font-family:'IBM Plex Mono', monospace !important;
    }

    /* Selectbox text containers */

    div[data-testid="stForm"] div[data-baseweb="select"] [data-testid="stMarkdownContainer"],
    div[data-testid="stForm"] div[data-baseweb="select"] p{
      color:#F1E8D6 !important;
    }

    /* Dropdown arrow */

    div[data-testid="stForm"] div[data-baseweb="select"] svg{
      fill:#F1E8D6 !important;
      color:#F1E8D6 !important;
    }

    /* Focus state */

    div[data-testid="stForm"] input:focus,
    div[data-testid="stForm"] div[data-baseweb="select"] > div:focus-within{
      border-bottom:1.5px solid var(--brass) !important;
      outline:none !important;
    }

    /* Number input +/- buttons */

    div[data-testid="stForm"] button{
      color:#F1E8D6 !important;
    }

    div[data-testid="stForm"] button svg{
      fill:#F1E8D6 !important;
    }


    /* ---------- Radio buttons ---------- */

    div[data-testid="stForm"] div[role="radiogroup"]{
      display:flex;
      gap:6px;
      flex-wrap:wrap;
    }

    div[data-testid="stForm"] div[role="radiogroup"] label{
      border:1px solid var(--rule);
      padding:5px 14px 4px 14px;
      border-radius:20px;
      margin-right:0 !important;
      opacity:1 !important;
      transition:all 0.15s ease;
    }

    div[data-testid="stForm"] div[role="radiogroup"] label:has(input:checked){
      background:var(--brass);
      border-color:var(--brass);
    }

    div[data-testid="stForm"] div[role="radiogroup"] label div p{
      font-family:'Inter', sans-serif !important;
      text-transform:none !important;
      letter-spacing:0 !important;
      font-size:0.86rem !important;
      color:var(--text-on-paper) !important;
      opacity:1 !important;
    }


    /* ---------- Ledger rules ---------- */

    hr.ledger-rule{
      border:none;
      border-top:1px dashed var(--rule);
      margin:6px 0 18px 0;
    }


    /* ---------- Submit button ---------- */

    div[data-testid="stForm"] button[kind="formSubmit"],
    .stButton > button{
      background:var(--brass) !important;
      color:#1c1408 !important;
      border:none !important;
      border-radius:2px !important;

      font-family:'IBM Plex Mono', monospace !important;
      letter-spacing:0.08em;
      text-transform:uppercase;
      font-size:0.76rem !important;
      padding:10px 22px !important;
      font-weight:600 !important;

      transition:
        transform 0.1s ease,
        background 0.15s ease;
    }

    div[data-testid="stForm"] button[kind="formSubmit"]:hover,
    .stButton > button:hover{
      background:var(--brass-2) !important;
      transform:translateY(-1px);
    }


    /* ---------- Decision slip ---------- */

    .slip-empty{
      font-family:'IBM Plex Mono', monospace;
      font-size:0.85rem;
      color:var(--text-on-paper);
      opacity:0.55;
      padding:40px 6px 60px 6px;
      line-height:1.7;
    }

    .slip-row{
      display:flex;
      justify-content:space-between;
      font-family:'IBM Plex Mono', monospace;
      font-size:0.82rem;
      padding:5px 0;
      border-bottom:1px dotted rgba(42,34,22,0.22);
    }

    .slip-row span.k{
      opacity:0.62;
    }

    .slip-row span.v{
      font-weight:600;
    }


    /* ---------- Approval / decline stamp ---------- */

    .stamp-wrap{
      position:relative;
      display:flex;
      align-items:center;
      justify-content:center;
      margin:18px 0 10px 0;
      height:128px;
    }

    @keyframes stampHit{
      0%{
        transform:scale(2.6) rotate(-14deg);
        opacity:0;
      }

      55%{
        transform:scale(0.94) rotate(-9deg);
        opacity:1;
      }

      75%{
        transform:scale(1.05) rotate(-11deg);
      }

      100%{
        transform:scale(1) rotate(-10deg);
        opacity:1;
      }
    }

    .stamp{
      font-family:'Fraunces', serif;
      font-weight:600;
      font-size:1.5rem;
      letter-spacing:0.06em;
      text-transform:uppercase;
      padding:14px 26px;
      border:4px double currentColor;
      border-radius:8px;
      transform:rotate(-10deg);
      animation:stampHit 0.5s cubic-bezier(.2,.8,.3,1.1);
      white-space:nowrap;
    }

    .stamp.approve{
      color:var(--sage);
    }

    .stamp.decline{
      color:var(--rust);
    }

    .risk-seal{
      font-family:'IBM Plex Mono', monospace;
      font-size:0.68rem;
      letter-spacing:0.1em;
      text-transform:uppercase;
      text-align:center;
      margin-top:2px;
      color:var(--text-on-paper);
      opacity:0.75;
    }

    .risk-seal b{
      color:var(--brass-2);
      opacity:1;
    }


    /* ---------- Expander ---------- */

    div[data-testid="stExpander"]{
      background:transparent;
      border:1px solid rgba(231,226,211,0.14);
      border-radius:4px;
    }

    div[data-testid="stExpander"] summary{
      font-family:'IBM Plex Mono', monospace;
      font-size:0.75rem;
      letter-spacing:0.08em;
      text-transform:uppercase;
      color:var(--text-dim) !important;
    }

    div[data-testid="stExpander"] summary p{
      color:var(--text-dim) !important;
    }


    /* ---------- Dataframe ---------- */

    div[data-testid="stDataFrame"]{
      border-radius:4px;
      overflow:hidden;
    }


    /* ---------- Footer note ---------- */

    .footnote{
      font-family:'IBM Plex Mono', monospace;
      font-size:0.68rem;
      color:var(--text-dim);
      opacity:0.6;
      margin-top:36px;
      line-height:1.6;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Data + model training (cached once per server process)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Reconciling the ledger — training the desk's models…")
def train_models():
    df = pd.read_csv("loan_data.csv")
    df = df.drop("Loan_ID", axis=1)

    for col in ["Gender", "Married", "Dependents", "Self_Employed"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(360).astype(int)
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0]).astype(int)
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

    df["Income_to_Loan_Ratio"] = (df["TotalIncome"] / df["LoanAmount"]).replace([np.inf, -np.inf], np.nan)
    df["Income_to_Loan_Ratio"] = df["Income_to_Loan_Ratio"].fillna(df["Income_to_Loan_Ratio"].median())
    df["Income_Per_Dependent"] = df["TotalIncome"] / (df["Dependents"] + 1)
    df["Monthly_Loan_Burden"] = (df["LoanAmount"] / df["Loan_Amount_Term"]).replace([np.inf, -np.inf], np.nan)
    df["Monthly_Loan_Burden"] = df["Monthly_Loan_Burden"].fillna(df["Monthly_Loan_Burden"].median())

    df = df.drop(["ApplicantIncome", "CoapplicantIncome"], axis=1)
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)
    df["Loan_Status"] = df["Loan_Status"].map({"N": 0, "Y": 1})

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]
    columns = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    grid = GridSearchCV(
        LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000),
        {"C": [0.01, 0.1, 1, 10, 100], "solver": ["liblinear", "lbfgs"]},
        cv=5,
        scoring="f1",
    )
    grid.fit(X_train_s, y_train)
    best_lr = grid.best_estimator_

    nb = GaussianNB().fit(X_train_s, y_train)
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=3,
        class_weight="balanced", random_state=42,
    ).fit(X_train, y_train)

    def metrics(model, Xte, yte):
        pred = model.predict(Xte)
        return dict(
            Accuracy=accuracy_score(yte, pred),
            Precision=precision_score(yte, pred),
            Recall=recall_score(yte, pred),
            F1=f1_score(yte, pred),
        )

    comparison = pd.DataFrame(
        {
            "Logistic Regression": metrics(best_lr, X_test_s, y_test),
            "Naive Bayes": metrics(nb, X_test_s, y_test),
            "Random Forest": metrics(rf, X_test, y_test),
        }
    ).T.round(3)

    importance = (
        pd.DataFrame({"Feature": columns, "Weight": best_lr.coef_[0]})
        .assign(AbsWeight=lambda d: d["Weight"].abs())
        .sort_values("AbsWeight", ascending=True)
        .tail(8)
    )

    return {
        "model": best_lr,
        "scaler": scaler,
        "columns": columns,
        "comparison": comparison,
        "importance": importance,
    }


def build_feature_row(raw, columns):
    row = pd.DataFrame([raw])
    row["Dependents"] = (
    row["Dependents"]
    .astype(str)
    .str.strip()
    .str.replace("3+", "3", regex=False)
)
row["Dependents"] = pd.to_numeric(row["Dependents"], errors="coerce").fillna(0).astype(int)
    row["TotalIncome"] = row["ApplicantIncome"] + row["CoapplicantIncome"]
    row["Income_to_Loan_Ratio"] = row["TotalIncome"] / row["LoanAmount"]
    row["Income_Per_Dependent"] = row["TotalIncome"] / (row["Dependents"] + 1)
    row["Monthly_Loan_Burden"] = row["LoanAmount"] / row["Loan_Amount_Term"]
    row = row.drop(["ApplicantIncome", "CoapplicantIncome"], axis=1)
    row = pd.get_dummies(row, columns=CAT_COLS)
    row = row.reindex(columns=columns, fill_value=0)
    return row


def risk_of(prob):
    if prob >= 0.80:
        return "Low Risk", "sage"
    if prob >= 0.60:
        return "Medium Risk", "brass"
    return "High Risk", "rust"


bundle = train_models()

# --------------------------------------------------------------------------
# Session state: file number + quick-fill presets
# --------------------------------------------------------------------------
if "file_no" not in st.session_state:
    st.session_state.file_no = random.randint(100000, 999999)

DEFAULTS = dict(
    gender="Male", married="Yes", dependents="0", education="Graduate",
    self_employed="No", app_income=5000, coapp_income=0, loan_amount=140,
    loan_term=360, credit_history="Yes", property_area="Semiurban",
)

PRESETS = {
    "strong": dict(
        gender="Female", married="Yes", dependents="0", education="Graduate",
        self_employed="No", app_income=9200, coapp_income=2100, loan_amount=110,
        loan_term=360, credit_history="Yes", property_area="Semiurban",
    ),
    "borderline": dict(
        gender="Male", married="No", dependents="1", education="Not Graduate",
        self_employed="Yes", app_income=3200, coapp_income=800, loan_amount=150,
        loan_term=360, credit_history="Yes", property_area="Rural",
    ),
    "risky": dict(
        gender="Male", married="Yes", dependents="3+", education="Not Graduate",
        self_employed="No", app_income=2100, coapp_income=0, loan_amount=180,
        loan_term=360, credit_history="No", property_area="Rural",
    ),
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v
    st.session_state.pop("result", None)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    f"""
<div class="eyebrow-row">
  <span>FILE NO. <span class="mark">{st.session_state.file_no}</span> &nbsp;·&nbsp; DESK 3</span>
  <span>LEDGERLINE <span class="mark">&amp;</span> BOND — LOAN OFFICE</span>
</div>
<div class="masthead">
  <h1>Will the loan <em>go through?</em></h1>
  <p class="sub">Fill out the applicant file the way a loan officer would. The desk runs it against
  614 past decisions and returns a probability, a risk grade, and its reasoning — before you ever
  reach a real underwriter.</p>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Quick-fill row
# --------------------------------------------------------------------------
st.markdown('<div class="quickfill">', unsafe_allow_html=True)
qcols = st.columns([1, 1, 1, 4])
with qcols[0]:
    if st.button("Strong file", key="preset_strong"):
        apply_preset("strong")
        st.rerun()
with qcols[1]:
    if st.button("Borderline file", key="preset_borderline"):
        apply_preset("borderline")
        st.rerun()
with qcols[2]:
    if st.button("Risky file", key="preset_risky"):
        apply_preset("risky")
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------------------------
# Main layout: applicant file (form) | decision slip
# --------------------------------------------------------------------------
left, right = st.columns([1.15, 1], gap="large")

with left:
    with st.form("application"):
        st.markdown('<div class="card-label">Section A</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Applicant File</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            gender = st.radio("Gender", ["Male", "Female"], key="gender", horizontal=True)
            married = st.radio("Married", ["Yes", "No"], key="married", horizontal=True)
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], key="dependents")
        with c2:
            education = st.radio("Education", ["Graduate", "Not Graduate"], key="education", horizontal=True)
            self_employed = st.radio("Self-employed", ["Yes", "No"], key="self_employed", horizontal=True)
            property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"], key="property_area")

        st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Section B</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Income &amp; Loan Terms</div>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            app_income = st.number_input(
                "Your monthly income ($)", min_value=0, step=100, key="app_income"
            )
            coapp_income = st.number_input(
                "Co-applicant's monthly income ($, 0 if none)", min_value=0, step=100, key="coapp_income"
            )
            credit_history = st.radio(
                "Clean credit history?", ["Yes", "No"], key="credit_history", horizontal=True
            )
        with c4:
            loan_amount = st.number_input(
                "Loan amount requested (in $000s)", min_value=1, step=5, key="loan_amount"
            )
            loan_term = st.selectbox(
                "Loan term (months)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], key="loan_term"
            )

        st.write("")
        submitted = st.form_submit_button("Submit application")

with right:
    st.markdown('<div class="ledger-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Section C</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Decision Slip</div>', unsafe_allow_html=True)

    if submitted:
        raw = dict(
            Gender=gender, Married=married, Dependents=dependents, Education=education,
            Self_Employed=self_employed, ApplicantIncome=app_income, CoapplicantIncome=coapp_income,
            LoanAmount=loan_amount, Loan_Amount_Term=loan_term,
            Credit_History=1 if credit_history == "Yes" else 0, Property_Area=property_area,
        )
        row = build_feature_row(raw, bundle["columns"])
        scaled = bundle["scaler"].transform(row)
        prob = bundle["model"].predict_proba(scaled)[0][1]
        decision = "approve" if prob >= 0.5 else "decline"
        risk_label, risk_color = risk_of(prob)
        st.session_state["result"] = dict(
            prob=prob, decision=decision, risk_label=risk_label, risk_color=risk_color,
            raw=raw, total_income=app_income + coapp_income,
        )

    result = st.session_state.get("result")

    if not result:
        st.markdown(
            """
            <div class="slip-empty">
            No decision on file yet.<br>
            Complete the applicant file on the left and press
            <b>Submit application</b> — the stamp falls here.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        prob = result["prob"]
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"family": "IBM Plex Mono", "size": 34, "color": "#2a2216"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b7a52", "tickfont": {"size": 10}},
                    "bar": {"color": "#B7863C", "thickness": 0.28},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 60], "color": "rgba(154,74,53,0.18)"},
                        {"range": [60, 80], "color": "rgba(183,134,60,0.18)"},
                        {"range": [80, 100], "color": "rgba(63,107,83,0.18)"},
                    ],
                },
            )
        )
        fig.update_layout(
            height=190,
            margin=dict(l=20, r=20, t=15, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "IBM Plex Mono"},
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        stamp_class = "approve" if result["decision"] == "approve" else "decline"
        stamp_text = "Loan Approved" if result["decision"] == "approve" else "Application Declined"
        st.markdown(
            f"""
            <div class="stamp-wrap"><div class="stamp {stamp_class}">{stamp_text}</div></div>
            <div class="risk-seal">Risk grade — <b>{result['risk_label']}</b></div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)
        r = result["raw"]
        rows = [
            ("Household income / mo", f"${result['total_income']:,.0f}"),
            ("Loan requested", f"${r['LoanAmount']*1000:,.0f}"),
            ("Term", f"{r['Loan_Amount_Term']} months"),
            ("Credit history", "Clean" if r["Credit_History"] == 1 else "Flagged"),
            ("Property area", r["Property_Area"]),
        ]
        for k, v in rows:
            st.markdown(
                f'<div class="slip-row"><span class="k">{k}</span><span class="v">{v}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Behind the decision
# --------------------------------------------------------------------------
st.write("")
with st.expander("Behind the decision — what the desk weighs, and how the models compare"):
    bcol1, bcol2 = st.columns([1.2, 1])

    with bcol1:
        st.markdown(
            "<div style='font-family:IBM Plex Mono; font-size:0.75rem; color:#c9c3ae; "
            "letter-spacing:0.05em; text-transform:uppercase; margin-bottom:8px;'>"
            "Logistic regression coefficients (top 8, by strength)</div>",
            unsafe_allow_html=True,
        )
        imp = bundle["importance"]
        colors = ["#3f6b53" if w > 0 else "#9a4a35" for w in imp["Weight"]]
        fig2 = go.Figure(
            go.Bar(
                x=imp["Weight"], y=imp["Feature"], orientation="h",
                marker_color=colors,
            )
        )
        fig2.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "IBM Plex Mono", "color": "#e7e2d3", "size": 11},
            xaxis={"gridcolor": "rgba(231,226,211,0.08)", "zerolinecolor": "rgba(231,226,211,0.25)"},
            yaxis={"gridcolor": "rgba(231,226,211,0.0)"},
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            "Green bars push toward approval, rust bars push toward decline. "
            "Credit history and household income typically dominate."
        )

    with bcol2:
        st.markdown(
            "<div style='font-family:IBM Plex Mono; font-size:0.75rem; color:#c9c3ae; "
            "letter-spacing:0.05em; text-transform:uppercase; margin-bottom:8px;'>"
            "Model comparison, held-out test set</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(bundle["comparison"], use_container_width=True)
        st.caption(
            "The desk's live decisions come from the tuned Logistic Regression model. "
            "Naive Bayes and Random Forest are shown for comparison only."
        )

st.markdown(
    """
    <div class="footnote">
    Ledgerline is a machine-learning demo trained on a public sample of 614 historical loan
    records. It is not a real credit decision, does not use your actual financial data, and
    should not be relied on for an actual lending decision.
    </div>
    """,
    unsafe_allow_html=True,
)
