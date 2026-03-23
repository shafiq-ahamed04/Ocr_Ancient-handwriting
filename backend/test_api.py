import requests

url = "http://localhost:8000/ocr/manuscript"
test_image_path = r"C:\final year Project\ml\dataset\gokul\image1.png"

with open(test_image_path, "rb") as f:
    files = {"file": ("image1.png", f, "image/png")}
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        
        data = response.json()
        print("Successfully hit OCR endpoint!")
        print("Detected text:")
        print(data.get("text"))
        print(f"Number of lines detected: {data.get('num_lines')}")
    except Exception as e:
        print(f"Error testing backend API: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Server response: {response.text}")
