import io
import warnings

warnings.filterwarnings("ignore")

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import streamlit as st

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Medical AI Clinical Platform",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "https://medical-ai-clinical-platform.onrender.com"

# ============================================================
# Session state helpers
# ============================================================
if "token" not in st.session_state:
    st.session_state.token = None
if "patient" not in st.session_state:
    st.session_state.patient = None
if "inference_result" not in st.session_state:
    st.session_state.inference_result = None
if "dicom_pixels" not in st.session_state:
    st.session_state.dicom_pixels = None


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def api_get(path: str, **kwargs) -> httpx.Response:
    return httpx.get(f"{API_BASE}{path}", headers=auth_headers(), timeout=60, **kwargs)


def api_post(path: str, **kwargs) -> httpx.Response:
    return httpx.post(f"{API_BASE}{path}", headers=auth_headers(), timeout=120, **kwargs)


# ============================================================
# Sidebar - Login
# ============================================================
with st.sidebar:
    st.title("🫁 Medical AI Platform")
    st.caption("Portfolio Demo — Not for Clinical Use")
    st.divider()

    if st.session_state.token is None:
        st.subheader("Login")
        username = st.text_input("Username", value="demo")
        password = st.text_input("Password", type="password", value="demo123")
        if st.button("Login", use_container_width=True):
            try:
                r = httpx.post(
                    f"{API_BASE}/auth/token",
                    data={"username": username, "password": password},
                    timeout=10,
                )
                if r.status_code == 200:
                    st.session_state.token = r.json()["access_token"]
                    st.success(f"Logged in as {username}")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            except Exception as e:
                st.error(f"Cannot reach API: {e}")
    else:
        st.success("Logged in")
        if st.button("Logout", use_container_width=True):
            st.session_state.token = None
            st.session_state.patient = None
            st.session_state.inference_result = None
            st.rerun()

    st.divider()
    st.caption("Demo credentials: demo / demo123")

# ============================================================
# Require login
# ============================================================
if st.session_state.token is None:
    st.info("Please log in using the sidebar to continue.")
    st.stop()

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Patient Lookup",
    "2. DICOM Inference",
    "3. FHIR Report",
    "4. Audit Log",
])

# ============================================================
# Tab 1: Patient Lookup
# ============================================================
with tab1:
    st.header("Patient Lookup")
    st.caption("Fetch a patient record from the HAPI FHIR R4 server.")

    col1, col2 = st.columns([2, 1])
    with col1:
        patient_id = st.text_input("FHIR Patient ID", value="90270587", placeholder="e.g. 90270587")
    with col2:
        st.write("")
        st.write("")
        lookup = st.button("Fetch Patient", use_container_width=True)

    if lookup and patient_id:
        with st.spinner("Fetching from HAPI FHIR..."):
            try:
                r = api_get(f"/fhir/patient/{patient_id}")
                if r.status_code == 200:
                    st.session_state.patient = r.json()
                    st.success("Patient found")
                elif r.status_code == 404:
                    st.error(f"Patient {patient_id} not found on HAPI server.")
                else:
                    st.error(f"Error {r.status_code}: {r.text[:200]}")
            except Exception as e:
                st.error(f"API error: {e}")

    if st.session_state.patient:
        p = st.session_state.patient
        st.subheader("Patient Demographics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Name", p.get("display_name", "N/A"))
        col2.metric("Gender", p.get("gender", "N/A").title())
        col3.metric("Date of Birth", p.get("birth_date", "N/A"))
        st.info(f"FHIR Reference: `{p.get('fhir_reference')}`")

# ============================================================
# Tab 2: DICOM Inference
# ============================================================
with tab2:
    st.header("DICOM Inference")
    st.caption("Upload a chest X-ray DICOM file. The AI will analyse it and write a FHIR DiagnosticReport.")

    if st.session_state.patient is None:
        st.warning("Please look up a patient first (Tab 1).")
    else:
        st.info(f"Linked patient: **{st.session_state.patient.get('display_name')}** "
                f"({st.session_state.patient.get('fhir_reference')})")

        uploaded = st.file_uploader("Upload DICOM (.dcm)", type=["dcm"])

        if uploaded:
            dicom_bytes = uploaded.read()

            # Preview the DICOM image
            try:
                ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
                arr = ds.pixel_array.astype(float)
                p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
                if p99 > p1:
                    arr = np.clip((arr - p1) / (p99 - p1), 0, 1)
                else:
                    arr = np.zeros_like(arr)
                st.session_state.dicom_pixels = arr
                st.subheader("DICOM Preview")
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(arr, cmap="gray", aspect="equal")
                ax.axis("off")
                ax.set_title(f"{getattr(ds, 'Modality', 'DICOM')} - {arr.shape[0]}x{arr.shape[1]}px", fontsize=10)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Could not preview DICOM: {e}")

            patient_fhir_id = st.session_state.patient["fhir_id"]

            if st.button("Run AI Inference", type="primary", use_container_width=True):
                with st.spinner("Running inference (300-500ms on CPU)..."):
                    try:
                        r = api_post(
                            "/infer/",
                            files={"file": (uploaded.name, dicom_bytes, "application/octet-stream")},
                            data={"patient_fhir_id": patient_fhir_id},
                        )
                        if r.status_code == 200:
                            st.session_state.inference_result = r.json()
                            st.success("Inference complete. FHIR report written.")
                        else:
                            st.error(f"Inference failed ({r.status_code}): {r.text[:300]}")
                    except Exception as e:
                        st.error(f"API error: {e}")

        if st.session_state.inference_result:
            result = st.session_state.inference_result
            st.divider()
            st.subheader("AI Findings")

            col1, col2, col3 = st.columns(3)
            col1.metric("Inference Time", f"{result['inference_time_ms']:.0f} ms")
            col2.metric("Model", result["model_id"].split("-")[1].upper() if "-" in result["model_id"] else result["model_id"])
            col3.metric("Audit Entry", f"#{result['audit_sequence']}")

            st.caption("**Top 5 Pathologies (AI confidence scores — requires radiologist review)**")
            for finding in result["top_findings"]:
                pct = finding["percentage"]
                color = "red" if pct > 50 else "orange" if pct > 35 else "green"
                st.markdown(
                    f"**{finding['name']}** — {pct:.1f}%"
                )
                st.progress(pct / 100)

            with st.expander("All 18 pathology scores"):
                for finding in result["all_findings"]:
                    st.text(f"{finding['name']:35s} {finding['percentage']:5.1f}%")

            st.caption(
                "⚠️ AI-assisted analysis only. All findings require radiologist review before clinical use."
            )

# ============================================================
# Tab 3: FHIR Report
# ============================================================
with tab3:
    st.header("FHIR DiagnosticReport")

    if st.session_state.inference_result is None:
        st.info("Run an inference first (Tab 2) to generate a report.")
    else:
        result = st.session_state.inference_result
        report_id = result["report_fhir_id"]
        fhir_url = f"http://hapi.fhir.org/baseR4/DiagnosticReport/{report_id}"

        st.success(f"Report written to HAPI FHIR: **DiagnosticReport/{report_id}**")
        st.markdown(f"[View raw FHIR JSON on HAPI]({fhir_url})")

        col1, col2 = st.columns(2)
        col1.metric("Report ID", report_id)
        col2.metric("Status", "preliminary")

        st.subheader("Report Details")
        st.json({
            "resourceType": "DiagnosticReport",
            "id": report_id,
            "status": "preliminary",
            "subject": {"reference": f"Patient/{result['patient_fhir_id']}"},
            "model_id": result["model_id"],
            "model_version": result["model_version"],
            "source_dicom_sha256": result["source_dicom_sha256"][:16] + "...",
            "weights_sha256": result["weights_sha256"],
            "top_findings": result["top_findings"],
        })

        st.caption(
            "Status is 'preliminary' — AI output pending radiologist review. "
            "Will not change to 'final' without qualified clinical sign-off."
        )

# ============================================================
# Tab 4: Audit Log
# ============================================================
with tab4:
    st.header("Audit Log")
    st.caption("Append-only hash-chained audit trail. Every inference is permanently recorded.")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Verify Chain Integrity", use_container_width=True):
            with st.spinner("Verifying..."):
                try:
                    r = api_get("/audit/verify")
                    if r.status_code == 200:
                        chain = r.json()
                        if chain["is_valid"]:
                            st.success(f"Chain intact — {chain['total_entries']} entries verified")
                        else:
                            st.error(f"CHAIN BROKEN at sequence {chain['first_broken_sequence']}: {chain['broken_reason']}")
                except Exception as e:
                    st.error(f"API error: {e}")

    with st.spinner("Loading audit events..."):
        try:
            r = api_get("/audit/?n=20")
            if r.status_code == 200:
                events = r.json()
                if events:
                    import pandas as pd
                    df = pd.DataFrame([{
                        "Seq": e["sequence"],
                        "Event": e["event_type"],
                        "Actor": e["actor"],
                        "Model": e["model_id"] or "-",
                        "Timestamp (UTC)": e["timestamp_utc"][:19].replace("T", " "),
                        "Hash (first 12)": e["entry_hash"][:12] + "...",
                    } for e in events])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No audit events yet.")
        except Exception as e:
            st.error(f"API error: {e}")

    st.divider()
    st.caption(
        "Regulatory note: Each row hash = SHA-256(row content + previous hash). "
        "Modification or deletion of any row breaks the chain and is detected by 'Verify Chain Integrity'. "
        "SQLite BEFORE triggers prevent UPDATE/DELETE at the database level (21 CFR Part 11 §11.10(e))."
    )
