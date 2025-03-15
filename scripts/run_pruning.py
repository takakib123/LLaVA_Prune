#!/usr/bin/env python
"""
CLI script for running pruning and inference with LLaVA models.
"""

import argparse
import os
from transformers import AutoProcessor
from llava_prune import iterative_pruning_and_inference


def main():
    parser = argparse.ArgumentParser(description="Prune LLaVA models and perform inference")
    parser.add_argument("--model-name", type=str, default="Aranya31/Derm-LLaVA-1.5-7b-conv2",
                        help="Name or path of the pre-trained model")
    parser.add_argument("--processor-id", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="ID of the processor to use")
    parser.add_argument("--processor-revision", type=str, default="a272c74",
                        help="Revision of the processor")
    parser.add_argument("--unimportance-orders", type=str, 
                        default="23,4,18,22,21,24,17,27,20,26,19,29,28,5",
                        help="Comma-separated list of layer indices to prune in order of unimportance")
    parser.add_argument("--prompt", type=str, default="What are the clinical features of rosacea?",
                        help="Text prompt to use")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image to use")
    parser.add_argument("--output", type=str, default="responses.csv",
                        help="Path to the output CSV file")

    args = parser.parse_args()
    
    # Parse unimportance orders
    unimportance_orders = list(map(int, args.unimportance_orders.split(',')))
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.processor_id,
        revision=args.processor_revision
    )
    
    # Run pruning and inference
    iterative_pruning_and_inference(
        args.model_name,
        processor,
        unimportance_orders,
        args.prompt,
        args.image,
        args.output
    )


if __name__ == "__main__":
    main()
