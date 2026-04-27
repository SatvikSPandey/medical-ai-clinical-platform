# Medical AI Clinical Platform

> **Portfolio Project — Senior AI Engineer | Siemens Healthineers Track**
> A production-grade clinical AI platform with DICOM handling, HL7 FHIR R4 integration, FDA-aligned audit controls, and chest pathology inference using DenseNet-121.

[![CI](https://github.com/SatvikSPandey/medical-ai-clinical-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/SatvikSPandey/medical-ai-clinical-platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/Docker-multi--stage-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![License](https://img.shields.io/badge/license-MIT-green)

## Live Demo

| Service | URL |
|---|---|
| Streamlit UI | https://medical-ai-clinical-platform-satvik.streamlit.app |
| FastAPI (Render) | https://medical-ai-clinical-platform.onrender.com/docs |

> **Note:** Render free tier spins down after inactivity. First request may take ~50s.

---

## Architecture

┌─────────────────────────────────────────────────────────┐
│                    Streamlit Cloud UI                    │
│  Patient Lookup │ DICOM Inference │ FHIR │ Audit Log    │
└──────────────────────────┬──────────────────────────────┘
│ HTTPS (JWT)
┌──────────────────────────▼──────────────────────────────┐
│                  FastAPI — Render Free                   │
│   /auth  │  /infer  │  /fhir  │  /audit  │  /health    │
└────┬─────────────┬──────────────┬──────────┬────────────┘
│             │              │          │
┌────▼────┐  ┌─────▼──────┐ ┌────▼────┐ ┌──▼──────────┐
│  DICOM  │  │ ML Inference│ │  FHIR   │ │ Audit Log   │
│ Parser  │  │ DenseNet-121│ │  R4     │ │ Hash-Chain  │
│ SHA-256 │  │ Grad-CAM++ │ │  Client │ │ SQLite      │
└─────────┘  └────────────┘ └─────────┘ └─────────────┘
│
┌───────────▼──────────┐
│   HAPI FHIR Server   │
│ (public R4 sandbox)  │
└──────────────────────┘

---

## Key Features

**DICOM Handling**
- Parser with SHA-256 provenance tracking per file
- Pixel normalisation pipeline (HU windowing, percentile stretch)
- 30 unit tests, 93.5% coverage

**ML Inference**
- torchxrayvision DenseNet-121 (NIH ChestX-ray14 weights)
- 18-class pathology detection (Pneumonia, Atelectasis, Cardiomegaly, etc.)
- Grad-CAM++ visual explainability — meets FDA AI/ML guidance for clinical decision support
- ~230ms inference on CPU

**HL7 FHIR R4**
- Patient reader from HAPI public sandbox
- DiagnosticReport writer — structured findings pushed back to FHIR server
- SNOMED-CT coded observations

**Compliance & Audit**
- Append-only hash-chained SQLite audit log (SHA-256 chain)
- SQLite BEFORE UPDATE/DELETE triggers — tamper-evident
- `verify_chain()` endpoint for integrity checks
- FDA SaMD Pre-Submission alignment doc included

**API & Auth**
- FastAPI with JWT Bearer auth
- 5 production endpoints: `/health`, `/auth/token`, `/infer/`, `/fhir/patient/{id}`, `/audit/`
- OpenAPI docs at `/docs`

**DevOps**
- Multi-stage Docker build (builder + runtime), CPU-only PyTorch (~1.8GB image)
- GitHub Actions CI: ruff lint + mypy typecheck + pytest (63 tests)
- Deployed to Render free tier

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | PyTorch 2.5, torchxrayvision, pytorch-grad-cam |
| API | FastAPI 0.115, Uvicorn, JWT (python-jose) |
| Medical | pydicom, HL7 FHIR R4 (requests) |
| Compliance | SQLite (hash-chained), FDA SaMD docs |
| UI | Streamlit, httpx, matplotlib |
| DevOps | Docker (multi-stage), GitHub Actions, Render |
| Testing | pytest, 63 tests, 93.5% DICOM coverage |

---

## Project Structure

medical-ai-clinical-platform/
├── api/                    # FastAPI app, routers, JWT auth
├── ml/                     # DenseNet-121 inference, Grad-CAM++, model registry
├── dicom_handler/          # DICOM parser, normaliser, SHA-256 provenance
├── fhir_client/            # FHIR R4 reader/writer
├── compliance/             # Hash-chained audit log, FDA docs
├── ui/                     # Streamlit clinical workflow UI
├── tests/                  # pytest suites (DICOM, audit, FHIR)
├── Dockerfile.api          # Multi-stage CPU Docker build
├── docker-compose.yml
└── .github/workflows/ci.yml

---

## Local Setup

```bash
git clone https://github.com/SatvikSPandey/medical-ai-clinical-platform
cd medical-ai-clinical-platform
python -m venv venv && venv\Scripts\activate   # Windows
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

```bash
# Run API
uvicorn api.main:app --reload

# Run UI (separate terminal)
streamlit run ui/app.py

# Run tests
pytest tests/ -q
```

```bash
# Docker
docker-compose up --build
```

**Demo credentials:** `demo / demo123` · `radiologist / rad456`

---

## Interview Notes

This project demonstrates production patterns required in clinical AI roles:

- **FDA alignment** — Grad-CAM++ explainability + append-only audit log map directly to FDA AI/ML SaMD guidance (2021) and EU AI Act Annex IV requirements for high-risk AI systems
- **FHIR R4** — industry standard for health data interoperability; DiagnosticReport resource follows IHE RAD profile structure
- **Hash-chained audit** — each log entry includes `SHA-256(previous_hash + entry_data)`, making retrospective tampering detectable — a requirement for 21 CFR Part 11 compliance
- **Multi-stage Docker** — builder stage installs gcc/g++ for compilation, runtime stage is minimal; non-root `appuser` for security

---

*Built as an independent freelance portfolio project. Not affiliated with any employer.*
---

## Author

**Satvik Pandey** — Senior AI Engineer | LLM Systems | Computer Vision | Data Engineering

- GitHub: [SatvikSPandey](https://github.com/SatvikSPandey)
- LinkedIn: [satvikpandey-433555365](https://linkedin.com/in/satvikpandey-433555365)
- Portfolio: [satvikspandey.netlify.app](https://satvikspandey.netlify.app)
- Email: satvikpan@gmail.com
