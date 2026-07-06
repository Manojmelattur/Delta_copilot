from auth import signed_get

response = signed_get("/v2/profile")
print(response)
