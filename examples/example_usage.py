"""
Example usage of the LLaVA pruning package.
"""

from transformers import AutoProcessor
from llava_prune import iterative_pruning_and_inference

def main():
    # Example configuration
    model_name = "Aranya31/Derm-LLaVA-1.5-7b-conv2"
    processor_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(
        processor_id,
        revision='a272c74'
    )
    
    unimportance_orders = [23, 4, 18, 22, 21, 24, 17, 27, 20, 26, 19, 29, 28, 5]
    prompt = "What are the clinical features of rosacea?"
    image_url = "./images/rosacea-70.jpg"
    output_csv = "responses.csv"
    
    # Run pruning and inference
    iterative_pruning_and_inference(
        model_name,
        processor,
        unimportance_orders,
        prompt,
        image_url,
        output_csv
    )

if __name__ == "__main__":
    main()
