"""
Pipeline functions for pruning and inference with LLaVA models.
"""

import csv
import os
from tqdm import tqdm
from .pruning import prune_llava_model
from .inference import perform_inference


def iterative_pruning_and_inference(model_name, processor, unimportance_orders, 
                                   prompt_text, image_url, output_csv):
    """
    Perform iterative pruning and inference with a LLaVA model.
    
    Args:
        model_name (str): The name or path of the pre-trained model.
        processor: The processor for the model.
        unimportance_orders (list): List of layer indices to prune in order of unimportance.
        prompt_text (str): The text prompt to use.
        image_url (str): URL or path to the image to use.
        output_csv (str): Path to the output CSV file.
        
    Returns:
        None: Results are saved to the output CSV file.
    """
    # Initialize CSV file
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as csvfile:
        fieldnames = ['pruned_layers', 'response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Original model performance
        model = prune_llava_model(model_name, [])
        response = perform_inference(model, processor, prompt_text, image_url)
        writer.writerow({
            'pruned_layers': 'None',
            'response': response
        })
        
        # Iterative pruning
        pruned_layers = []
        for layer in tqdm(unimportance_orders):
            pruned_layers.append(layer)
            model = prune_llava_model(model_name, pruned_layers)
            response = perform_inference(model, processor, prompt_text, image_url)
            writer.writerow({
                'pruned_layers': ','.join(map(str, pruned_layers)),
                'response': response
            })
            
    print(f"Results saved to {output_csv}")
