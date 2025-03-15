import torch
from PIL import Image
import requests
from io import BytesIO


def perform_inference(model, processor, prompt_text, image_url):
    """
    Perform inference using a LLaVA model.
    
    Args:
        model: The LLaVA model to use for inference.
        processor: The processor for the model.
        prompt_text (str): The text prompt to use.
        image_url (str): URL or path to the image to use.
        
    Returns:
        str: The generated response.
    """
    # Load image
    if image_url.startswith(("http://", "https://")):
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_url)
    
    # Process inputs
    inputs = processor(
        prompt_text,
        image,
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )
    
    # Process output
    response = processor.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from the response
    response = response.replace(prompt_text, "").strip()
    
    return response
