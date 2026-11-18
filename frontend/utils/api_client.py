"""
FinGuard AI — Frontend API Client

Handles HTTP communication between the Streamlit frontend and the FastAPI backend,
including JWT token management, automatic error handling, and file downloads.
"""

import os
from typing import Any, Dict, List, Optional
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")


class APIClient:
    """Wrapper for FinGuard AI FastAPI endpoints."""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def _get_headers(self, token: Optional[str] = None) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate and obtain JWT access token."""
        url = f"{self.base_url}/auth/login"
        resp = requests.post(
            url,
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def register(self, username: str, email: str, password: str, role: str = "analyst") -> Dict[str, Any]:
        """Register a new user account."""
        url = f"{self.base_url}/auth/register"
        resp = requests.post(
            url,
            json={"username": username, "email": email, "password": password, "role": role},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def get_me(self, token: str) -> Dict[str, Any]:
        """Fetch current authenticated user profile."""
        url = f"{self.base_url}/auth/me"
        resp = requests.get(url, headers=self._get_headers(token), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_dashboard_stats(self, token: str) -> Dict[str, Any]:
        """Fetch high-level KPI summary for the dashboard."""
        url = f"{self.base_url}/dashboard/stats"
        resp = requests.get(url, headers=self._get_headers(token), timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_transactions(
        self,
        token: str,
        page: int = 1,
        size: int = 50,
        vendor_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        risk_level: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Fetch paginated transactions with scores and filters."""
        url = f"{self.base_url}/transactions"
        params: Dict[str, Any] = {"page": page, "size": size}
        if vendor_id:
            params["vendor_id"] = vendor_id
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        if risk_level and risk_level != "ALL":
            params["risk_level"] = risk_level
        if min_amount is not None:
            params["min_amount"] = min_amount
        if max_amount is not None:
            params["max_amount"] = max_amount

        resp = requests.get(url, headers=self._get_headers(token), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_transaction_detail(self, token: str, transaction_id: str) -> Dict[str, Any]:
        """Fetch full transaction detail, vendor profile, and anomaly score."""
        url = f"{self.base_url}/transactions/{transaction_id}"
        resp = requests.get(url, headers=self._get_headers(token), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def take_transaction_action(
        self, token: str, transaction_id: str, action: str, comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """Post analyst action: approve, escalate, or flag_audit."""
        url = f"{self.base_url}/transactions/{transaction_id}/action"
        resp = requests.post(
            url,
            headers=self._get_headers(token),
            json={"action": action, "comment": comment},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def get_vendors(
        self,
        token: str,
        page: int = 1,
        size: int = 50,
        risk_tier: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch paginated vendors."""
        url = f"{self.base_url}/vendors"
        params: Dict[str, Any] = {"page": page, "size": size}
        if risk_tier and risk_tier != "ALL":
            params["risk_tier"] = risk_tier
        if search:
            params["search"] = search
        resp = requests.get(url, headers=self._get_headers(token), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_vendor_risk(self, token: str, vendor_id: str) -> Dict[str, Any]:
        """Fetch vendor risk profile and historical scores."""
        url = f"{self.base_url}/vendors/{vendor_id}/risk"
        resp = requests.get(url, headers=self._get_headers(token), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def score_single_transaction(self, token: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run on-demand prediction and SHAP explanation."""
        url = f"{self.base_url}/predict"
        resp = requests.post(
            url,
            headers=self._get_headers(token),
            json=payload,
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()

    def get_audit_log(
        self, token: str, page: int = 1, size: int = 50, action: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch audit log items (admin only)."""
        url = f"{self.base_url}/reports/audit-log"
        params: Dict[str, Any] = {"page": page, "size": size}
        if action:
            params["action"] = action
        resp = requests.get(url, headers=self._get_headers(token), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def download_csv_export(
        self, token: str, date_from: Optional[str] = None, date_to: Optional[str] = None
    ) -> bytes:
        """Download anomalies CSV export."""
        url = f"{self.base_url}/reports/anomalies/export"
        params: Dict[str, Any] = {}
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        resp = requests.get(url, headers=self._get_headers(token), params=params, timeout=30)
        resp.raise_for_status()
        return resp.content

    def download_pdf_report(
        self,
        token: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        risk_levels: Optional[List[str]] = None,
    ) -> bytes:
        """Generate and download PDF risk report."""
        url = f"{self.base_url}/reports/pdf"
        body = {
            "date_from": date_from,
            "date_to": date_to,
            "risk_levels": risk_levels or ["HIGH", "CRITICAL"],
            "max_rows": 300,
        }
        resp = requests.post(url, headers=self._get_headers(token), json=body, timeout=45)
        resp.raise_for_status()
        return resp.content

    # -----------------------------------------------------------------------
    # O2C (Order-to-Cash) Credit Management APIs
    # -----------------------------------------------------------------------

    def get_o2c_customers(self, token: str, page: int = 1, size: int = 50) -> Dict[str, Any]:
        url = f"{self.base_url}/o2c/customers"
        resp = requests.get(url, headers=self._get_headers(token), params={"page": page, "size": size}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_o2c_customer_detail(self, token: str, customer_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/o2c/customers/{customer_id}"
        resp = requests.get(url, headers=self._get_headers(token), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_o2c_exposure(self, token: str, customer_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/o2c/exposure/{customer_id}"
        resp = requests.get(url, headers=self._get_headers(token), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def evaluate_o2c_credit_check(
        self, token: str, customer_id: str, order_value: float, order_number: Optional[str] = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/o2c/credit-check"
        payload = {"customer_id": customer_id, "order_value": order_value, "order_number": order_number}
        resp = requests.post(url, headers=self._get_headers(token), json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_o2c_ar_aging(self, token: str, customer_id: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/o2c/ar-aging"
        params = {"customer_id": customer_id} if customer_id else {}
        resp = requests.get(url, headers=self._get_headers(token), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_o2c_payment_behaviour(self, token: str, customer_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/o2c/payment-behaviour"
        resp = requests.get(url, headers=self._get_headers(token), params={"customer_id": customer_id}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_o2c_anomalies(
        self, token: str, customer_id: Optional[str] = None, risk_category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/o2c/anomalies"
        params = {}
        if customer_id:
            params["customer_id"] = customer_id
        if risk_category and risk_category != "ALL":
            params["risk_category"] = risk_category
        resp = requests.get(url, headers=self._get_headers(token), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def score_o2c_customer(self, token: str, customer_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/o2c/predict"
        resp = requests.post(url, headers=self._get_headers(token), json={"customer_id": customer_id}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # -----------------------------------------------------------------------
    # R2R (Record-to-Report) APIs
    # -----------------------------------------------------------------------

    def get_r2r_gl_accounts(
        self, token: str, company_code: Optional[str] = None, account_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/r2r/gl-accounts"
        params = {}
        if company_code:
            params["company_code"] = company_code
        if account_type:
            params["account_type"] = account_type
        resp = requests.get(url, headers=self._get_headers(token), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_r2r_journals(
        self,
        token: str,
        page: int = 1,
        size: int = 50,
        company_code: Optional[str] = None,
        journal_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/r2r/journals"
        params = {"page": page, "size": size}
        if company_code:
            params["company_code"] = company_code
        if journal_type and journal_type != "ALL":
            params["journal_type"] = journal_type
        resp = requests.get(url, headers=self._get_headers(token), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_r2r_journal_detail(self, token: str, journal_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/r2r/journals/{journal_id}"
        resp = requests.get(url, headers=self._get_headers(token), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def validate_r2r_journal(self, token: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/r2r/journals/validate"
        resp = requests.post(url, headers=self._get_headers(token), json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_r2r_reconciliations(
        self, token: str, company_code: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/r2r/reconciliations"
        params = {}
        if company_code:
            params["company_code"] = company_code
        if status and status != "ALL":
            params["status"] = status
        resp = requests.get(url, headers=self._get_headers(token), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_r2r_intercompany(self, token: str, matching_status: Optional[str] = None) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/r2r/intercompany"
        params = {"matching_status": matching_status} if matching_status else {}
        resp = requests.get(url, headers=self._get_headers(token), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_r2r_close_status(self, token: str, company_code: str = "1000", fiscal_period: int = 12) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/r2r/close-status"
        params = {"company_code": company_code, "fiscal_period": fiscal_period}
        resp = requests.get(url, headers=self._get_headers(token), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_r2r_anomalies(self, token: str, risk_category: Optional[str] = None) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/r2r/anomalies"
        params = {"risk_category": risk_category} if risk_category and risk_category != "ALL" else {}
        resp = requests.get(url, headers=self._get_headers(token), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def score_r2r_journal(self, token: str, journal_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/r2r/predict"
        resp = requests.post(url, headers=self._get_headers(token), json={"journal_id": journal_id}, timeout=15)
        resp.raise_for_status()
        return resp.json()
