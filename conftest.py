"""Root-level pytest configuration.

Adds the project root to sys.path so tests can import top-level packages
(dicom_handler, ml, fhir_client, compliance, api) without installing the
project as a package. This is the standard pytest convention for
src-less project layouts.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
