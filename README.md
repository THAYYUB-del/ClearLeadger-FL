# ClearLedger-FL

**Blockchain-Based Secure and Explainable Federated Model Aggregation for Healthcare Data Sharing**

ClearLedger-FL is a working federated learning platform simulating three independent hospital nodes (`Hospital A`, `Hospital B`, and `Hospital C`) that collaboratively train a diagnostic logistic regression classifier on the Wisconsin Breast Cancer Diagnostic (WDBC) dataset without exchanging raw patient records.

> **Educational Scope Notice**: This software is an educational machine learning and cryptographic-audit engineering platform. It is not a certified medical device and does not claim HIPAA compliance or clinical diagnostic approval.

---

## Core Features

- **Federated Learning (FedAvg)**: Deterministic dataset splitting (Balanced IID and skewed Non-IID clinical case mixes), isolated local logistic regression training per hospital, and sample-weighted Federated Averaging (`w_global = Σ (n_k / N) w_k`).
- **Two-Layer Security Center**:
  - *Layer 1 (Cryptographic Authentication)*: HMAC-SHA256 signature verification (`hmac.compare_digest` / `crypto.timingSafeEqual`) using server-side hospital secret keys. Catches invalid signatures and forged senders.
  - *Layer 2 (Malicious Update Screening)*: Independent parameter L2-norm bounds and directional cosine-similarity screening. Explicitly demonstrates that a valid HMAC key proves identity, not honesty — catching exaggerated scaling and label-flipping poisoning attacks.
- **Local Hash-Linked Audit Ledger**: SHA-256 hash-chained audit trail persisted in SQLite with full chain integrity verification and a controlled, non-destructive tampering test sandbox. (Accurately scoped as a local cryptographic hash-linked ledger, not a decentralized consensus blockchain.)
- **Explainable AI (SHAP)**: Global feature importance and per-patient prediction explanations via `shap.LinearExplainer` (Python backend) and exact Interventional Linear SHAP (`φ_j(x) = w_j (x_j − E[X_j])`).
- **Model Performance Analytics & CSV Export**: Live Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, and side-by-side comparison of Centralized vs. Individual Hospital vs. Federated Global models, exportable to CSV.

---

## Technologies

- **Frontend**: React 19, Vite, TypeScript, Tailwind CSS, Recharts, Lucide React
- **Python Backend (`backend/`)**: Python 3.10+, FastAPI, Pydantic, scikit-learn, NumPy, Pandas, SHAP, SQLite, `hmac`/`hashlib`
- **Integrated Node Runtime (`server.ts`, `server/`)**: Express, TypeScript, `node:sqlite`, `node:crypto` (provides full API & mathematical parity for single-command web preview environments without Python pip packages)

---

## Installation & Running

### Option A: Windows One-Click Launch (Python FastAPI + React)

1. Copy `.env.example` to `.env` and set your local HMAC secret placeholders.
2. Run:
   ```bat
   run.bat
   ```
   This starts the FastAPI backend on `http://127.0.0.1:8000` and the React frontend on `http://localhost:5173`.

### Option B: Manual Setup (Python FastAPI + React)

**Backend (Python FastAPI):**
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate  |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

**Frontend (React + Vite):**
```bash
npm install
VITE_API_PROXY=http://127.0.0.1:8000 npm run dev:frontend
```

### Option C: Single-Server Full-Stack Mode (Node.js 22+)

```bash
npm install
npm run dev
```
Runs both the API and React UI on `http://localhost:3000`.

---

## Running Tests

**Python Security, Ledger & SQLite Tests (stdlib):**
```bash
python -m unittest backend/tests/test_security_and_ledger.py
```

**Full Python FastAPI + scikit-learn + SHAP Integration Tests:**
```bash
pytest backend/tests/test_api_and_ml.py
```

**TypeScript Engine Unit Tests:**
```bash
npm test
```

---

## Project Folder Structure

```text
ClearLedger-FL/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI routes (19 REST endpoints)
│   │   ├── blockchain/     # SHA-256 hash-linked audit ledger & tamper sandbox
│   │   ├── explainability/ # Real SHAP LinearExplainer calculations
│   │   ├── federated/      # scikit-learn WDBC loader, IID/Non-IID split, NumPy FedAvg
│   │   ├── models/         # Pydantic request schemas
│   │   ├── security/       # HMAC-SHA256 auth & Layer 2 malicious-update validator
│   │   └── services/       # Parameterized SQLite DB & federated coordinator
│   ├── tests/              # unittest & pytest suites
│   ├── main.py             # FastAPI ASGI entrypoint
│   └── requirements.txt
├── src/                    # React + TypeScript + Tailwind + Recharts UI
│   ├── charts/             # Reusable Recharts clinical visualizations
│   ├── components/         # Shared enterprise UI components
│   ├── layouts/            # Deep navy sidebar & responsive top bar shell
│   ├── pages/              # 7 interactive application pages
│   └── services/           # Typed API client
├── server/                 # Integrated Node/SQLite parity backend engine
├── server.ts               # Express + Vite server entrypoint
├── docs/                   # System architecture documentation
├── results/                # Local SQLite database directory (gitignored .db files)
├── .env.example
├── .gitignore
├── run.bat
└── README.md
```
