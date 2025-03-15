import torch
from PIL import Image
import requests
from io import BytesIO

def perform_inference(model, processor, prompt_text, image_url):
    """
    Perform inference on a given prompt and image.

    Args:
        model: The pruned LLAVA model loaded on GPU.
        processor: The AutoProcessor associated with the model.
        prompt_text: A string containing the user prompt.
        image_url: URL of the image to process.

    Returns:
        The generated response as a string.
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image"},
            ],
        },
    ]

    # Apply chat template to the conversation
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Load and preprocess the image
    # raw_image = Image.open(requests.get(image_url, stream=True).raw)
    raw_image = Image.open(image_url)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

    # Perform inference
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    # Decode and return the result
    return processor.decode(output[0], skip_special_tokens=True)
