# ClearLedger-FL Architecture Overview

## 1. Data Isolation Boundary
- The Wisconsin Breast Cancer Diagnostic (WDBC) dataset (569 samples, 30 cytological features) is split into an 80% training pool (455 records) and a 20% held-out evaluation set (114 records).
- Z-score standardization is fitted strictly on the training set to prevent evaluation leakage.
- Training records are partitioned into three disjoint hospital nodes (`Hospital A`, `Hospital B`, `Hospital C`) under either Balanced (IID) or skewed clinical case-mix (Non-IID) rules.
- Raw patient records remain encapsulated inside each hospital partition. Only signed parameter vectors (`w ∈ R^30`, `b ∈ R`) and sample counts cross the network boundary to the aggregator.

## 2. Two-Layer Update Verification
1. **Layer 1 — Cryptographic Authentication (HMAC-SHA256)**:
   - Verifies that the update was signed using the hospital's server-side shared secret key and was not modified in transit (`hmac.compare_digest` / `crypto.timingSafeEqual`).
   - Blocks invalid HMAC digests and forged-sender impersonation attempts.
2. **Layer 2 — Malicious Update Validation**:
   - Operates independently after HMAC succeeds, because a valid cryptographic key only proves sender identity — not update honesty.
   - Screens for exaggerated parameter scaling via L2 delta-norm thresholds (`||Δw||_2`) and label-flipping poisoning via directional cosine similarity against peer consensus.

## 3. Local Hash-Linked Audit Ledger
- Every experiment initialization, accepted hospital update, and FedAvg aggregation is appended to an SQLite-backed SHA-256 hash chain where `current_hash = SHA-256(canonical_record + previous_hash)`.
- Verification recomputes every digest from Genesis (`0000...0000`) through the latest record.
- The tampering demonstration clones the ledger in memory so users can inspect exact hash-mismatch detection without corrupting the live experiment history.
