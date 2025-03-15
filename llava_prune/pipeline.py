"""
Pipeline functions for pruning and inference with LLaVA models.
"""

import csv
import os
from tqdm import tqdm
from .pruning import prune_llava_model
from .inference import perform_inference


def iterative_pruning_and_inference(model_name, processor, unimportance_orders, prompt_text, image_url, output_csv):
    """
    Iteratively prune the model, perform inference, and save responses.

    Args:
        model_name: Name of the model to load.
        processor: The AutoProcessor associated with the model.
        unimportance_orders: List of lists containing unimportance orders for pruning.
        prompt_text: A string containing the user prompt.
        image_url: URL of the image to process.
        output_csv: Path to the CSV file where results will be saved.
    """
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["num_of_layers","Order", "Params" , "response"])

        for i in range(1, len(unimportance_orders) + 1):
            current_order = unimportance_orders[:i]
            print(f"Running pruning for unimportance order: {current_order}")

            # Prune the model
            model_pruned = prune_llava_model(model_name, current_order)
            total_params = 0
            total_params = sum(p.numel() for p in model_pruned.parameters() if p.requires_grad)

            # Perform inference
            t = time.time()
            response = perform_inference(model_pruned, processor, prompt_text, image_url)
            
            print(f"Inference time {time.time() - t:.5f} sec. Response for pruning {current_order}: {response}, ")

            # Save results to CSV
            num_layers_remaining = 32- i
            writer.writerow([num_layers_remaining, current_order, total_params , response])

            # Clear GPU memory
            del model_pruned
            del total_params
            torch.cuda.empty_cache()
            gc.collect()
