# 🏢 MGC Aurora Heights — AI Sales Assistant & Sprint Task

**Candidate:** Mujahid Abbas  
**Role:** AI Developer & Engineer  
**Stack:** Python 3.10+, Flask, scikit-learn, SQLite, TF-IDF  
**Repository:** [https://github.com/iamujahidabbas572/MGC-Real-Estate-Ai-sales-assistant](https://github.com/iamujahidabbas572/MGC-Real-Estate-Ai-sales-assistant)

---

## 🎥 Video Demo

Watch the full 4-part end-to-end demonstration (Document Assistant Q&A, SQL Schema & Deduplication queries, ML Lead Scoring baseline, and the live Flask Web Application):

[![Watch the Loom Demo Video](https://cdn.loom.com/sessions/thumbnails/3ad0e2e23c994a21b29bc4d7c1b58a27-with-play.gif)](https://www.loom.com/share/3ad0e2e23c994a21b29bc4d7c1b58a27)

> **[▶️ Click Here to Watch the Live Loom Demo Video](https://www.loom.com/share/3ad0e2e23c994a21b29bc4d7c1b58a27)**  
> *(Alternative direct repository MP4 file: [`All 4 parts demo.mp4`](./All%204%20parts%20demo.mp4))*

---

## ⚡ Quick Start

```bash
# 1. Install required dependencies
pip install -r requirements.txt

# 2. Run Part 1 — Document Assistant (demo of all 5 test questions)
python part1_document_assistant.py --demo

# 3. Run Part 2 — SQL Schema & Queries (loads CSV into SQLite, runs both queries)
python part2_run_queries.py

# 4. Run Part 3 — Train the lead scoring model (prints F1 score & saves model)
python part3_lead_scoring.py

# 5. Run Part 4 — Web interface (open http://127.0.0.1:5000)
python app.py
```

> **Note:** Run Part 3 before Part 4 so `lead_scoring_model.pkl` is generated for the web app.

---

## 📄 Part 1 — Grounded Document Assistant

An AI document assistant that answers plain-English salesperson queries grounded **strictly** in the three MGC documents (`brochure`, `price list`, and `booking policy FAQ`). Runs 100% locally with zero external API key requirements.

### Hard Cases Handling Matrix:

| Test Question | Why It's Here | Behaviour & Source Citation |
|---|---|---|
| *"What's the base price of a 2-bed in Block B?"* | Straight lookup | **PKR 22,425,000** (1,150 sq ft @ 19,500/sq ft) — *Source: Price List (Apr 2025)* |
| *"What's the total for a Margalla-facing corner unit on floor 15, 2-bed Block B?"* | Cumulative stacked premiums | Base (PKR 26,855,000) + Floor 15 (+4%) + Corner (+3%) + Margalla (+6%) = **+13% total premium (PKR 3,491,150)** $\rightarrow$ **Total: PKR 30,346,150** — *Source: Price List* |
| *"What's the transfer fee?"* | **Document conflict** | ⚠️ **Explicitly flags disagreement**: Price List states 2%, Booking Policy states 2.5%. Recommends sales agent verify with sales manager — *Sources: Price List & Booking FAQ* |
| *"What's the rental yield on a 1-bed?"* | Missing from docs | 🛑 **Refusal per policy**: Cites Booking FAQ clause that MGC does not publish verbal yield projections; redirects inquiry to the Marketing Manager |
| *"Who is the anchor tenant?"* | Explicitly unconfirmed | ⚠️ **Uncertainty flag**: Notes discussions are ongoing and no tenant is confirmed as of the brochure release date — *Source: Brochure (Mar 2025)* |

---

## 🗄️ Part 2 — Database: Schema & Queries

### Schema Design (`schema.sql`):
- Single `leads` table with strong SQL data types (`VARCHAR`, `TIMESTAMP`, `NUMERIC`, `BOOLEAN`).
- Primary key on `lead_id`.
- **Deduplication Defense:** `UNIQUE` constraint on `crm_record_hash`. When two sales agents attempt to insert the same lead, the database rejects the duplicate, forcing CRM deduplication.

### Queries (`queries.sql`):
1. **Conversion Rate by Lead Source** ($\ge 200$ leads, ordered best first):
   * *Referral:* 13.01% (95 / 730)
   * *Walk-in:* 10.33% (63 / 610)
   * *Facebook Ads:* 6.76% (160 / 2366)
   * *Google Search:* 6.58% (96 / 1460)
   * *WhatsApp Campaign:* 6.39% (35 / 548)
   * *Property Portal:* 5.96% (108 / 1812)
   * *Instagram:* 5.46% (55 / 1007)
   * *Billboard:* 4.26% (12 / 282)
   * *Expo Stall:* 2.90% (10 / 345)
2. **Duplicate Detection Query:**
   * Uses a self-join `ON a.crm_record_hash = b.crm_record_hash AND a.lead_id < b.lead_id` to cleanly surface the 160 duplicate pairs (e.g. `MGC-104974` and `MGC-104974-B`).

---

## 🤖 Part 3 — ML: Lead Scoring Baseline

### Data Decisions:

#### 1. Columns Dropped:
* `token_amount_received_pkr`: **CRITICAL TARGET LEAKAGE DROP**. A lead that has already paid a token has already converted. Keeping this feature gives artificial 99%+ accuracy that fails completely on fresh, incoming leads.
* `lead_id` & `crm_record_hash`: System identifiers with zero predictive value.
* `created_at`: Raw timestamp bypassed for this rapid baseline.

#### 2. Data Cleaning & Imputation:
* `city`: Normalized inconsistent CRM spellings (`ISB`/`ISLAMABAD` $\rightarrow$ `Islamabad`, `khi`/`KARACHI` $\rightarrow$ `Karachi`, `Rwp` $\rightarrow$ `Rawalpindi`, etc.).
* `bedrooms`: 39% missing; imputed using median (`2.0`).
* `budget_pkr_lac`, `first_response_minutes`, `agent_experience_years`: Imputed with median.
* `area`: 5% missing; labeled as `'Unknown'`.
* Removed 160 duplicate leads using `crm_record_hash`.

### Metric Selection & Results:
* **Model:** Random Forest Classifier (`n_estimators=100`, `max_depth=12`, `class_weight='balanced'`)
* **Primary Metric: F1 Score (Converted Class) = 0.1481**
* **Why F1?** The dataset has a severe class imbalance (**only 6.9% positive conversion rate**). Standard accuracy is deceptive (a naive model predicting "Not Converted" for everyone achieves 93.1% accuracy while finding zero buyers). F1 balances precision (preventing wasted sales calls) and recall (capturing true prospective buyers).

---

## 🌐 Part 4 — Web Interface

A clean, functional single-page Flask web application (`app.py` + `templates/index.html`):
1. **Document Assistant Tab:** Input plain-English inquiries $\rightarrow$ returns grounded answers with exact source document citations.
2. **Lead Scorer Tab:** Input lead criteria (source, budget, calls, visits, etc.) $\rightarrow$ returns instantaneous conversion probability percentage with color-coded priority tier (`🟢 High`, `🟡 Medium`, `🔴 Low`).

---

## 📁 Repository Structure

```
├── README.md                        # Documentation & submission overview
├── requirements.txt                 # Python dependencies
├── docs/                            # 3 MGC source documents
│   ├── 01_mgc_aurora_heights_brochure.md
│   ├── 02_price_list_payment_plan.md
│   └── 03_booking_policy_faq.md
├── leads.csv                        # CRM leads dataset (9,160 rows)
├── part1_document_assistant.py      # Part 1: Grounded Q&A Assistant
├── schema.sql                       # Part 2: SQL schema with UNIQUE deduplication
├── queries.sql                      # Part 2: SQL queries
├── part2_run_queries.py             # Part 2: SQLite execution script
├── part3_lead_scoring.py            # Part 3: ML model training & evaluation
├── lead_scoring_model.pkl           # Part 3: Exported model artifact
├── app.py                           # Part 4: Flask web application
├── templates/
│   └── index.html                   # Part 4: Web UI frontend
└── All 4 parts demo.mp4             # 1-minute video demo of all 4 parts
```
