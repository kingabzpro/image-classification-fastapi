import requests

# Base URL for the API
BASE_URL = "http://localhost:8000"


def health_check():
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    print("Health Check Response:")
    print(response.json())


def get_metadata():
    url = f"{BASE_URL}/metadata"
    response = requests.get(url)
    print("Metadata Response:")
    print(response.json())


def get_class_names():
    url = f"{BASE_URL}/class_names"
    response = requests.get(url)
    print("Class Names Response:")
    print(response.json())


def predict_image(file_path):
    url = f"{BASE_URL}/predict"
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
    print("Predict Response:")
    print(response.json())


def predict_with_confidence(file_path):
    url = f"{BASE_URL}/predict_with_confidence"
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
    print("Predict with Confidence Response:")
    print(response.json())


if __name__ == "__main__":
    # Perform a health check
    health_check()

    # Get model metadata
    get_metadata()

    # Get class names
    get_class_names()

    # Provide the path to a sample image to test prediction endpoints
    sample_image_path = "data/sample/cat.png"  # Change this to your local image path

    # Predict the image class (single prediction)
    predict_image(sample_image_path)

    # Predict the image class with top-3 confidence scores
    predict_with_confidence(sample_image_path)
