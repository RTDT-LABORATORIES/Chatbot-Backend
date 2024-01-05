import os
import requests


def setup():
    url = os.environ["BLB_API_OPENAPI_SPEC_URL"]
    response = requests.get(url)

    if response.status_code == 200:
        with open("blb_openapi_spec.json", "w") as file:
            file.write(response.text)
    else:
        print(f"Failed to download file: status code {response.status_code}")
