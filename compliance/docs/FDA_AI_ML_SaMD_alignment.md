# FDA AI/ML Action Plan and SaMD Alignment Document

**Project:** Medical AI Clinical Platform (Project 18)  
**Version:** 0.1.0  
**Date:** 2026-04-27  
**Status:** Draft - For Portfolio Demonstration  

## Disclaimer

This document maps platform controls to FDA AI/ML guidance. This is a portfolio demonstration project, not a cleared medical device.

---

## Regulatory Context

The FDA published its AI/ML-Based Software as a Medical Device Action Plan (January 2021) and the draft Predetermined Change Control Plan (PCCP) guidance (2023) to address the unique challenges of adaptive AI/ML in clinical settings. Key principles include:

1. **Transparency** - patients and clinicians must understand what AI is doing
2. **Traceability** - every prediction must be traceable to a specific model version
3. **Real-World Performance Monitoring** - model behavior must be monitorable post-deployment
4. **Good Machine Learning Practice (GMLP)** - engineering rigor throughout the ML lifecycle

---

## Control Mapping

### Principle 1 - Transparency and Explainability

**FDA Guidance:** Transparency to users regarding the AI/ML device function and output. Clinicians must be able to understand the basis for AI recommendations.

**Implementation:**
- `ml/gradcam.py` - Grad-CAM++ heatmaps show which image regions drove each pathology prediction
- `GradCAMResult` carries `pathology`, `heatmap`, `model_record`, and `source_sha256` - fully traceable explainability artifact
- All FHIR DiagnosticReports are explicitly framed: "AI-ASSISTED ANALYSIS (requires radiologist review)" - the AI role is never obscured
- FHIR report `status: preliminary` (never `final`) - prevents AI output from being misread as clinician-verified

### Principle 2 - Traceability and Model Versioning

**FDA Guidance:** Each version of an AI/ML device must be clearly identified. Changes to the algorithm must be controlled and documented.

**Implementation (`ml/model_registry.py`):**
- `ModelRecord` captures: `model_id`, `model_name`, `version`, `framework`, `weights_sha256`, `weights_size_bytes`, `output_classes`, `input_shape`, `input_value_range`, `registered_at_utc`
- `weights_sha256` is a SHA-256 of the actual weights file bytes - not just a version string. If the file changes, the hash changes.
- `verify_weights_unchanged()` runs before every inference - real-time tamper detection
- Every `INFERENCE_COMPLETE` audit entry records the `model_id` and `weights_sha256` - every prediction is permanently linked to its exact model version

### Principle 3 - Good Machine Learning Practice (GMLP)

**FDA Guidance:** Use of appropriate data management, feature engineering, model training, evaluation, and documentation practices.

**Implementation:**
- **Data provenance:** `ParsedDicom.source_sha256` is a SHA-256 of the raw `.dcm` file bytes, computed at parse time and carried through the entire pipeline into the audit log and FHIR report
- **Preprocessing documentation:** `NormalizationParams` captures every transformation applied to the pixel data (slope, intercept, photometric correction, clip percentiles, clipped min/max) - fully reproducible
- **Model selection:** torchxrayvision DenseNet-121 trained on NIH/PadChest/CheXpert/MIMIC - peer-reviewed, citable (Cohen et al. 2022, arxiv:2111.00595)
- **Input validation:** `_normalize_to_xrv()` explicitly bridges our [0,1] normalizer output to the model expected [-1024,1024] range with documented linear transformation
- **Output framing:** Confidence threshold (30%) is documented and auditable; scores are sigmoid probabilities, not calibrated clinical probabilities - distinction is documented in code and report text

### Principle 4 - Real-World Performance Monitoring

**FDA Guidance:** Ability to monitor AI/ML device performance in the real world after deployment.

**Implementation:**
- `AuditEventType.INFERENCE_COMPLETE` entries accumulate over time and can be queried for drift analysis
- `verify_chain()` provides integrity verification of the entire historical record
- The `source_sha256` field enables tracing any prediction back to its exact input file
- `inference_time_ms` is recorded per prediction - performance regression is detectable
- Model tamper detection (`MODEL_TAMPER_DETECTED` event type) provides a specific alert for the highest-risk monitoring scenario

### PCCP Alignment - Predetermined Change Control

**FDA Draft Guidance:** A PCCP describes planned modifications to an AI/ML device and the methodology for implementing them in a controlled manner.

**Implementation:**
- The `model_id` convention (`torchxrayvision-densenet121-all-1.4.0`) encodes source library and version - a model upgrade produces a new `model_id`, creating a new `ModelRecord`, and a new audit chain entry
- `output_classes` is stored in the `ModelRecord` - if a new model version adds or removes pathologies, this is captured automatically
- `requirements.txt` with exact pinned versions constitutes a software bill of materials (SBOM) - a PCCP requirement for software changes

---

## Evidence Artifacts

| Artifact | Location | FDA Principle |
|---|---|---|
| Grad-CAM explainability | `ml/gradcam.py` | Transparency |
| Model registry | `ml/model_registry.py` | Traceability |
| Audit log | `compliance/audit.py` | Traceability + Monitoring |
| NormalizationParams | `dicom_handler/normalizer.py` | GMLP data provenance |
| DiagnosticReport framing | `fhir_client/writer.py` | Transparency |
| Pinned requirements | `requirements.txt` | GMLP + PCCP |
| DICOM SHA-256 | `dicom_handler/parser.py` | GMLP data provenance |

---

## Honest Limitations

This platform does not constitute:
- A submission-ready PCCP document (which requires clinical evidence and formal risk analysis)
- A 510(k) predicate comparison or De Novo request
- Clinical validation of model performance on a representative patient population
- A formal GMLP assessment per FDA draft guidance

The model (torchxrayvision DenseNet-121) is used for demonstration. Production deployment of an AI diagnostic tool requires prospective clinical validation, analytical validation, and regulatory submission appropriate to the intended use and risk classification.
