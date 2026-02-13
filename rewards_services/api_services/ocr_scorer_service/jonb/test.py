import sys
import os
import requests
from PIL import Image

# Add common utils path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../common')))
from utils import create_payload, parse_response

def test_ocr_scorer():
    # Configuration
    service_url = "http://127.0.0.1:18082/"

    # Image and prompt paths
    image_path = "/inspire/ssd/project/deepgen/liruihang-253108100075/workspace/deepgen/DeepGen-RL/temp/deepgen/deepgen_rl/1211_deepgen/1227_train_deepgen_t2i_ocr_qz_n16_onlydm/20260106_1515/training_samples/step_1/step_1_cuda0_0.0000_0.jpg"
    prompt_path = "/inspire/ssd/project/deepgen/liruihang-253108100075/workspace/deepgen/DeepGen-RL/temp/deepgen/deepgen_rl/1211_deepgen/1227_train_deepgen_t2i_ocr_qz_n16_onlydm/20260106_1515/training_samples/step_1/step_1_cuda0.txt"

    # Load image
    print(f"Loading image from: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    image = Image.open(image_path)
    print(f"Image size: {image.size}, mode: {image.mode}")

    # Load prompt
    print(f"Loading prompt from: {prompt_path}")
    if not os.path.exists(prompt_path):
        print(f"Error: Prompt file not found at {prompt_path}")
        return
    with open(prompt_path, 'r') as f:
        prompt = f.read().strip()
    print(f"Prompt: {prompt}")

    # Create payload
    images = [image]
    prompts = [prompt]
    payload = create_payload(images, prompts)
    print(f"Payload size: {len(payload)} bytes")

    # Send request
    print(f"\nSending request to {service_url}...")
    try:
        response = requests.post(service_url, data=payload, timeout=60)
        print(f"Response status code: {response.status_code}")

        # Parse response
        result = parse_response(response.content)
        print(f"\nResponse result:")
        print(result)

        if "scores" in result:
            print(f"\nOCR Scores: {result['scores']}")
        if "error" in result:
            print(f"\nError: {result['error']}")

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to service at {service_url}")
        print("Make sure the OCR Scorer service is running.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_ocr_scorer()
