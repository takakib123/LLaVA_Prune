import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import gc
import csv
import copy
import time


def prune_llava_model(model_name, unimportance_order):
    """
    Prune the LLAVA model based on unimportance order.

    Args:
        model_name: Name of the model to load.
        unimportance_order: List of layer indices to prune.

    Returns:
        The pruned model.
    """
    model_orig = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )


    def copy_weight(model, model_orig, list_pruned_blocks):
        connect_info = {}
        connect_info["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        connect_info["model.norm.weight"] = "model.norm.weight"
        connect_info["lm_head.weight"] = "lm_head.weight"

        vision_tower_orig = model_orig.vision_tower.state_dict()
        vision_tower_pruned = model.vision_tower.state_dict()

        for key in vision_tower_pruned.keys():
            if key in vision_tower_orig:
                vision_tower_pruned[key].copy_(vision_tower_orig[key])
            else:
                print(f"Warning: {key} not found in model_orig.vision_tower")
        print("Vision tower weights copied successfully.")

        mm_projector_orig = model_orig.multi_modal_projector.state_dict()
        mm_projector_pruned = model.multi_modal_projector.state_dict()

        for key in mm_projector_pruned.keys():
            if key in mm_projector_orig:
                mm_projector_pruned[key].copy_(mm_projector_orig[key])
            else:
                print(f"Warning: {key} not found in model_orig.multi_modal_projector")
        print("Multi-modal projector weights copied successfully.")

        k = 0
        for k_orig in range(model_orig.config.text_config.__getattribute__("num_hidden_layers")):
            if k_orig in list_pruned_blocks:  # Skip pruned blocks
                continue

            connect_info[f"model.layers.{k}."] = f"model.layers.{k_orig}."
            print(f"original model.layers.{k_orig} --> pruned model.layers.{k}")
            k += 1

        print(f" ** excluded blocks {list_pruned_blocks}")

        t0 = time.perf_counter()
        for k in model.language_model.state_dict().keys():
            flag = 0
            k_orig = k

            for prefix_key in connect_info.keys():
                if k.startswith(prefix_key):
                    flag = 1
                    k_orig = k_orig.replace(prefix_key, connect_info[prefix_key])
                    break

            if flag == 1:
                print(f"** forced COPY {k_orig} -> {k}")
                model.language_model.state_dict()[k].copy_(model_orig.language_model.state_dict()[k_orig])

        print(f"copy time --- {(time.perf_counter()-t0):.1f} sec")

        return model

    config = copy.deepcopy(model_orig.config)
    print(f"# blocks before pruning: {config.text_config.num_hidden_layers}")
    num_pruned_blocks = len(unimportance_order)
    config.text_config.__setattr__(
        "num_hidden_layers", (config.text_config.num_hidden_layers - num_pruned_blocks)
    )

    model_pruned = LlavaForConditionalGeneration(config).to("cpu")
    model_pruned = copy_weight(
        model_pruned, model_orig, unimportance_order[:num_pruned_blocks]
    )

    model_pruned.half()
    model_pruned.to("cuda")
    gc.collect()

    return model_pruned

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
            response = perform_inference(model_pruned, processor, prompt_text, image_url)
            print(f"Response for pruning {current_order}: {response}")

            # Save results to CSV
            num_layers_remaining = 32- i
            writer.writerow([num_layers_remaining, current_order, total_params , response])

            # Clear GPU memory
            del model_pruned
            del p 
            del total_params
            torch.cuda.empty_cache()
            gc.collect()

# Example usage
# model_name = "Aranya31/Derm-LLaVA-1.5-7b-conv2"
# processor = AutoProcessor.from_pretrained(model_name)
# unimportance_orders = [1, 2, 3]
# prompt = "What is in this photo?"
# image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# output_csv = "responses.csv"
# iterative_pruning_and_inference(model_name, processor, unimportance_orders, prompt, image_url, output_csv)
