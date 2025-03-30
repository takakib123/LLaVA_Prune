import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
import csv

# Model initialization module
def initialize_model(model_id="llava-hf/llava-1.5-7b-hf"):
    """Initialize the LLAVA model and processor."""
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

# Image processing module
def load_image(image_path):
    """Load an image from the specified path."""
    return Image.open(image_path)

# Question generation module
def generate_questions():
    """Return the list of predefined questions."""
    return [
        "What is the name of the disease?",
        "What is the treatment of the disease?",
        "What is the clinical feature of the disease?"
    ]

# Inference module
def process_question(model, processor, image, question):
    """Process a single question for an image and return the response."""
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image"}
        ]
    }]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    generated_ids = output[0][inputs['input_ids'].shape[1]:]
    return processor.decode(generated_ids, skip_special_tokens=True)

# CSV handling module
def write_to_csv(output_csv, data):
    """Write data to CSV file with predefined headers."""
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'disease_name', 'treatment', 'clinical_feature'])
        for row in data:
            writer.writerow(row)

# Main processing module
def process_image_folder(image_folder, output_csv, model, processor, questions):
    """Process all images in the folder and generate responses."""
    results = []
    
    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, image_file)
            image = load_image(image_path)
            
            responses = [process_question(model, processor, image, q) for q in questions]
            results.append([image_file] + responses)
    
    write_to_csv(output_csv, results)
    return results
def generate_and_get_csv():
    """Main function to orchestrate the image analysis process."""
    image_folder = '/kaggle/input/selective-dermnet-for-llm/test_merged_selective_resized/test_merged_selective_resized/known_39'
    output_csv = 'evaluation_dataset.csv'
    
    # Initialize model and processor
    model, processor = initialize_model()
    
    # Get questions
    questions = generate_questions()
    
    # Process images and generate CSV
    process_image_folder(image_folder, output_csv, model, processor, questions)
    
    print(f"Evaluation dataset has been saved to {output_csv}")
