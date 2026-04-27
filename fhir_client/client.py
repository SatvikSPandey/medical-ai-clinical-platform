"""FHIR R4 HTTP client.

Thin async wrapper over httpx for communicating with a FHIR R4 server.
All FHIR HTTP operations (GET, POST, PUT) go through this class.

Design principles:
- One class, one responsibility: HTTP transport only.
  No FHIR resource construction here â€” that is reader.py / writer.py.
- All methods are async so the FastAPI layer (Phase 6) can await them
  without blocking the event loop during network I/O.
- Errors are wrapped in FHIRClientError with enough context to produce
  a useful audit log entry.
- The base URL and timeout are injected at construction time so tests
  can point at a mock server without patching global state.
"""

from __future__ import annotations

from typing import Any, cast

import httpx

FHIR_JSON_MEDIA_TYPE = "application/fhir+json"

DEFAULT_HEADERS = {
    "Accept": FHIR_JSON_MEDIA_TYPE,
    "Content-Type": FHIR_JSON_MEDIA_TYPE,
}


class FHIRClientError(Exception):
    """Raised when a FHIR HTTP operation fails.

    Carries the HTTP status code (if available) so callers can
    distinguish 404 Not Found from 500 Server Error.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FHIRClient:
    """Async FHIR R4 REST client.

    Usage:
        async with FHIRClient("http://hapi.fhir.org/baseR4") as client:
            patient = await client.get("Patient", "12345")
            report_id = await client.post("DiagnosticReport", report_dict)
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
    ) -> None:
        """
        Args:
            base_url: FHIR server base URL e.g. "http://hapi.fhir.org/baseR4".
                      Trailing slash is stripped automatically.
            timeout: HTTP request timeout in seconds (default 30).
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> FHIRClient:
        self._client = httpx.AsyncClient(
            headers=DEFAULT_HEADERS,
            timeout=self._timeout,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _url(self, resource_type: str, resource_id: str | None = None) -> str:
        """Build a FHIR resource URL."""
        if resource_id:
            return f"{self._base_url}/{resource_type}/{resource_id}"
        return f"{self._base_url}/{resource_type}"

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise FHIRClientError(
                "FHIRClient must be used as an async context manager. "
                "Use: async with FHIRClient(...) as client:"
            )
        return self._client

    async def get_metadata(self) -> dict[str, Any]:
        """Fetch the server CapabilityStatement (connectivity check).

        Returns:
            Parsed JSON dict of the CapabilityStatement.

        Raises:
            FHIRClientError: On HTTP error or timeout.
        """
        client = self._require_client()
        try:
            r = await client.get(f"{self._base_url}/metadata")
            r.raise_for_status()
            return cast(dict[str, Any], r.json())
        except httpx.HTTPStatusError as e:
            raise FHIRClientError(
                f"FHIR metadata request failed: {e}", status_code=e.response.status_code
            ) from e
        except httpx.RequestError as e:
            raise FHIRClientError(f"FHIR metadata connection error: {e}") from e

    async def get(
        self, resource_type: str, resource_id: str
    ) -> dict[str, Any]:
        """Read a single FHIR resource by type and ID.

        Args:
            resource_type: e.g. "Patient", "DiagnosticReport".
            resource_id: Server-assigned resource ID.

        Returns:
            Parsed JSON dict of the resource.

        Raises:
            FHIRClientError: On 404, 5xx, or network error.
        """
        client = self._require_client()
        url = self._url(resource_type, resource_id)
        try:
            r = await client.get(url)
            r.raise_for_status()
            return cast(dict[str, Any], r.json())
        except httpx.HTTPStatusError as e:
            raise FHIRClientError(
                f"GET {url} failed with HTTP {e.response.status_code}",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            raise FHIRClientError(f"GET {url} connection error: {e}") from e

    async def search(
        self,
        resource_type: str,
        params: dict[str, str],
    ) -> dict[str, Any]:
        """Search for FHIR resources matching given parameters.

        Args:
            resource_type: e.g. "Patient", "DiagnosticReport".
            params: FHIR search parameters e.g. {"family": "Smith", "_count": "5"}.

        Returns:
            Parsed JSON Bundle dict.

        Raises:
            FHIRClientError: On HTTP error or network error.
        """
        client = self._require_client()
        url = self._url(resource_type)
        try:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return cast(dict[str, Any], r.json())
        except httpx.HTTPStatusError as e:
            raise FHIRClientError(
                f"SEARCH {url} failed with HTTP {e.response.status_code}",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            raise FHIRClientError(f"SEARCH {url} connection error: {e}") from e

    async def post(
        self,
        resource_type: str,
        resource: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a new FHIR resource (HTTP POST).

        Args:
            resource_type: e.g. "DiagnosticReport".
            resource: FHIR resource as a dict (will be JSON-serialized).

        Returns:
            The created resource dict (server assigns the ID).

        Raises:
            FHIRClientError: On HTTP error or network error.
        """
        client = self._require_client()
        url = self._url(resource_type)
        try:
            r = await client.post(url, json=resource)
            r.raise_for_status()
            return cast(dict[str, Any], r.json())
        except httpx.HTTPStatusError as e:
            raise FHIRClientError(
                f"POST {url} failed with HTTP {e.response.status_code}: {e.response.text[:200]}",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            raise FHIRClientError(f"POST {url} connection error: {e}") from e
