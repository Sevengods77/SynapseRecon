import requests

data = {
    "fragment_data": [
        {"piece_index": 1, "direction_hint": "Move right and down to row 3, column 4"},
        {"piece_index": 2, "direction_hint": "Move left to row 1, column 1"}
    ]
}

try:
    response = requests.post("http://localhost:8000/generate_steps", json=data)
    print("Status:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
