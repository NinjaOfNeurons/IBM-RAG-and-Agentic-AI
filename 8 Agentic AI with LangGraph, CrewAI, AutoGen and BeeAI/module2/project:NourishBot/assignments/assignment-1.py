import requests
import base64
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

# Setup credentials
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    # api_key = "<YOUR_API_KEY>"  # Normally you'd put an API key here
)
client = APIClient(credentials)

# Load test images
url_image_1 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5uo16pKhdB1f2Vz7H8Utkg/image-1.png'
url_image_2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fsuegY1q_OxKIxNhf6zeYg/image-2.png'
url_image_3 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/KCh_pM9BVWq_ZdzIBIA9Fw/image-3.png'
url_image_4 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/VaaYLw52RaykwrE3jpFv7g/image-4.png'

image_urls = [url_image_1, url_image_2, url_image_3, url_image_4]

# Encode all images to base64
print("Encoding images...")
encoded_images = []
for url in image_urls:
    encoded_images.append(base64.b64encode(requests.get(url).content).decode("utf-8"))
print(f"Successfully encoded {len(encoded_images)} images.\n")

# Setup model
model_id = "meta-llama/llama-3-2-90b-vision-instruct"
project_id = "skills-network"
params = TextChatParameters()

model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params
)

# ============================================
# TODO: COMPLETE THIS FUNCTION
# ============================================
def generate_model_response(encoded_image, user_query, assistant_prompt="You are a helpful assistant. Answer the following user query in 1 or 2 sentences: "):
    """
    Sends an image and a query to the model and retrieves the response.

    Parameters:
    - encoded_image (str): Base64-encoded image string.
    - user_query (str): The user's question about the image.
    - assistant_prompt (str): Optional prompt to guide the model's response.

    Returns:
    - str: The model's response for the given image and query.
    """
    
    # TODO: Step 1 - Create the messages object
    # The messages should be a list containing a dictionary with:
    # - "role": "user"
    # - "content": a list with two items:
    #     1. A text object with the combined assistant_prompt and user_query
    #     2. An image_url object with the base64-encoded image
    
    messages = [
        {"role": "user",
        "content":[
            {
                "type": "text",
                "text": assistant_prompt + user_query
        },
        {
            "type": "image_url",
            "image_url":{
                "url":"data:image/jpeg;base64," + encoded_image,
                } 

                }
            ]
        }
    ]
    
    # TODO: Step 2 - Send the request to the model
    # Use model.chat() with the messages parameter
    
    response = model.chat(messages=messages)  # Replace with actual API call
    
    # TODO: Step 3 - Extract and return the response content
    # The response format is: response['choices'][0]['message']['content']
    
    return response['choices'][0]['message']['content']


# ============================================
# TODO: RUN TESTS
# ============================================

print("="*50)
print("TESTING generate_model_response FUNCTION")
print("="*50)

# Test 1: Basic Image Description
print("\n=== Test 1: Basic Image Description ===")
test1_query = "Describe this image"
# Call generate_model_response with encoded_images[0] and test1_query
# Store result in test1_response
test1_response = generate_model_response(encoded_images[0], test1_query)

print(f"Query: {test1_query}")
print(f"Response: {test1_response}\n")

# Test 2: Specific Object Detection
print("=== Test 2: Specific Object Detection ===")
test2_query = "What is the person wearing?"
# Call generate_model_response with encoded_images[1] and test2_query
test2_response = generate_model_response(encoded_images[1], test2_query)

print(f"Query: {test2_query}")
print(f"Response: {test2_response}\n")

# Test 3: Scene Analysis
print("=== Test 3: Scene Analysis ===")
test3_query = "What weather condition is shown in this image?"
# Call generate_model_response with encoded_images[2] and test3_query
test3_response = generate_model_response(encoded_images[2], test3_query)

print(f"Query: {test3_query}")
print(f"Response: {test3_response}\n")

# Test 4: Text Recognition
print("=== Test 4: Text Recognition ===")
test4_query = "What is the serving size listed on this label?"
# Call generate_model_response with encoded_images[3] and test4_query
test4_response = generate_model_response(encoded_images[3], test4_query)

print(f"Query: {test4_query}")
print(f"Response: {test4_response}\n")

# Final message
print("="*50)
print("All tests completed! ✓")
print("="*50)

