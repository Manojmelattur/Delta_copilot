# =============================================================================
# auth.py - HMAC-SHA256 Authentication for Delta Exchange API
# =============================================================================

import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import requests

from config import API_KEY, API_SECRET, BASE_URL


def generate_signature(secret, message):
    """Generate HMAC-SHA256 signature."""
    message = bytes(message, "utf-8")
    secret = bytes(secret, "utf-8")
    h = hmac.new(secret, message, hashlib.sha256)
    return h.hexdigest()


def get_headers(method, path, query_string="", payload=""):
    """
    Build signed request headers.
    Signature covers: method + timestamp + path + query_string + payload
    query_string must include the leading '?' if present.
    Note: Signature expires in 5 seconds — keep system clock synced.
    """
    timestamp = str(int(time.time()))
    signature_data = method + timestamp + path + query_string + payload
    signature = generate_signature(API_SECRET, signature_data)

    return {
        "api-key": API_KEY,
        "timestamp": timestamp,
        "signature": signature,
        "User-Agent": "python-rest-client",
        "Content-Type": "application/json",
    }


def signed_get(path, params=None):
    """Authenticated GET request."""
    query_string = ""
    if params:
        query_string = "?" + urlencode(params)  # FIX: include leading '?'

    headers = get_headers("GET", path, query_string=query_string)
    url = BASE_URL + path + (query_string if query_string else "")  # FIX: no double '?'

    try:
        response = requests.get(url, headers=headers, timeout=(3, 27))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
        raise


def signed_delete(path, params=None):
    """Authenticated DELETE request."""
    body = ""
    query_string = ""

    if params:
        import json

        body = json.dumps(params, separators=(",", ":"))

    headers = get_headers("DELETE", path, query_string="", payload=body)
    url = BASE_URL + path

    try:
        response = requests.delete(url, headers=headers, data=body, timeout=(3, 27))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
        raise


def signed_post(path, payload: dict):
    """Authenticated POST request."""
    body = json.dumps(payload, separators=(",", ":"))  # compact, no spaces
    headers = get_headers("POST", path, query_string="", payload=body)
    url = BASE_URL + path

    try:
        response = requests.post(url, headers=headers, data=body, timeout=(3, 27))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
        raise


def public_get(path, params=None):
    """Unauthenticated GET request (for public endpoints like candles)."""
    url = BASE_URL + path
    response = requests.get(url, params=params, timeout=(3, 27))
    response.raise_for_status()
    return response.json()
