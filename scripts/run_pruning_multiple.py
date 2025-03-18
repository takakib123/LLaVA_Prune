# run_pruning_multiple.py
#!/usr/bin/env python
"""
CLI script for running inference with LLaVA models without pruning.
"""

import argparse
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from llava_prune.inference import perform_inference


def main():
    parser = argparse.ArgumentParser(description="Perform inference with LLaVA models")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name or path of the pre-trained model")
    parser.add_argument("--processor-id", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="ID of the processor to use")
    parser.add_argument("--processor-revision", type=str, default="a272c74",
                        help="Revision of the processor")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt to use")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image to use")

    args = parser.parse_args()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.processor_id,
        revision=args.processor_revision
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Run inference
    response = perform_inference(
        model,
        processor,
        args.prompt,
        args.image
    )
    
    print("\nModel Response:")
    print("--------------")
    print(response)


if __name__ == "__main__":
    main()
