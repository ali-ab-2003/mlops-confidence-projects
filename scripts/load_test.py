import requests
import random
import time
import numpy as np

url = "http://localhost:8000/predict"

for i in range(50):
    payload = {
        "features": [random.random() for _ in range(30)]
    }
    Generate the feature vector

    r = requests.post(url, json=payload)
    print(r.status_code)

    time.sleep(0.5)
