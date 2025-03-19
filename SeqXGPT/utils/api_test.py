import requests
import msgpack

url = "http://localhost:20098/inference"

# Data formatted correctly for Mosec
data = {"text": "This is a test", "do_generate": False}
packed_data = msgpack.packb(data)

headers = {
    "Content-Type": "application/msgpack"
}

response = requests.post(url, data=packed_data, headers=headers)

print("Status Code:", response.status_code)
print("Response:", msgpack.unpackb(response.content) if response.content else "No Content")