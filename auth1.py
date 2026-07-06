import hashlib
import hmac
import time

import requests

API_KEY = "JKRVuRbKjbLKnn1OpuTMEUM2w4N1Mj"
API_SECRET = "NwvgN73CWs83RV2BCdmREjiArQDEJR60CJBiTBEFPDXuNs33L7U43RH6ytPz"
BASE_URL = "https://cdn-ind.testnet.deltaex.org"


def generate_signature(secret, message):
    return hmac.new(
        bytes(secret, "utf-8"), bytes(message, "utf-8"), hashlib.sha256
    ).hexdigest()


method = "GET"
timestamp = str(int(time.time()))
path = "/v2/orders"
query_string = "?product_id=27&state=open"
payload = ""

signature = generate_signature(
    API_SECRET, method + timestamp + path + query_string + payload
)

headers = {
    "api-key": API_KEY,
    "timestamp": timestamp,
    "signature": signature,
    "Content-Type": "application/json",
}

resp = requests.get(
    f"{BASE_URL}{path}",
    params={"product_id": 27, "state": "open"},
    headers=headers,
    timeout=10,
)

print(f"Status : {resp.status_code}")
print(f"Body   : {resp.text[:300]}")
