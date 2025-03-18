#!/usr/bin/env python
"""
CLI script for pruning a LLaVA model and saving the pruned model to disk.
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM
from llava_prune.pruning import prune_llava_model


def main():
    parser = argparse.ArgumentParser(description="Prune a LLaVA model and save it")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name or path of the pre-trained model")
    parser.add_argument("--unimportance-orders", type=str, required=True,
                        help="Comma-separated list of layer indices to prune in order of unimportance")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the pruned model")
    parser.add_argument("--model-suffix", type=str, default="pruned",
                        help="Suffix to add to the pruned model name")
    parser.add_argument("--fp16", action="store_true",
                        help="Save model in FP16 precision")

    args = parser.parse_args()
    
    # Parse unimportance orders
    unimportance_orders = list(map(int, args.unimportance_orders.split(',')))
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prune the model
    print(f"Pruning model {args.model_name} with unimportance orders: {unimportance_orders}")
    pruned_model = prune_llava_model(args.model_name, unimportance_orders)
    
    # Create a model name that includes the pruning information
    model_base_name = args.model_name.split('/')[-1]
    pruned_layers_str = '_'.join(map(str, unimportance_orders))
    pruned_model_name = f"{model_base_name}_{args.model_suffix}_layers_{pruned_layers_str}"
    output_path = os.path.join(args.output_dir, pruned_model_name)
    
    # Save the pruned model
    print(f"Saving pruned model to {output_path}")
    pruned_model.save_pretrained(
        output_path,
        torch_dtype=torch.float16 if args.fp16 else None
    )
    
    print(f"Pruned model saved successfully.")
    print(f"You can load this model with: --model-name {output_path}")


if __name__ == "__main__":
    main()
