import time
from webbrowser import get

import requests

# from market_data import get_product_id

# --- Check public IP ---
try:
    # print(f"Product Id : {get_product_id('BTCUSD')}")
    ip = requests.get("https://api.ipify.org", timeout=5).text.strip()
    print(f"Your public IP : {ip}")
except Exception as e:
    print(f"IP check failed: {e}")

# --- Estimate clock skew using response headers ---
try:
    local_before = int(time.time())
    resp = requests.get(
        "https://cdn-ind.testnet.deltaex.org/v2/products?page_size=1", timeout=5
    )
    local_after = int(time.time())

    # Server time is in the Date header (HTTP standard)
    server_date = resp.headers.get("Date", "not found")
    print(f"Server Date header : {server_date}")
    print(f"Local timestamp    : {local_before}")
    print(f"HTTP Status        : {resp.status_code}")

    # Parse server time from header
    from email.utils import parsedate_to_datetime

    server_dt = parsedate_to_datetime(server_date)
    server_ts = int(server_dt.timestamp())
    skew = local_before - server_ts
    print(f"Clock skew (seconds): {skew}  {'<<< PROBLEM' if abs(skew) > 5 else '(OK)'}")

except Exception as e:
    print(f"Skew check failed: {e}")
