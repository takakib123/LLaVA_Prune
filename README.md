# Pruning LLAVA Model for Efficient Inference

## Features
- Prunes the LLAVA model based on a given unimportance order.
- Runs inference on pruned models to assess their performance.
- Iteratively prunes different sets of layers and saves responses.
- Saves inference results in a CSV file with the number of remaining layers and responses.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install torch transformers pillow requests
```

## Usage

### 1. Pruning and Inference
The script iteratively prunes the LLAVA model and performs inference on an image.

#### Example Code
```python
from transformers import AutoProcessor

model_name = "Aranya31/Derm-LLaVA-1.5-7b-conv2"
unimportance_orders = [1, 2, 3]
prompt = "What is in this photo?"
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
processor = AutoProcessor.from_pretrained(model_name)

iterative_pruning_and_inference(model_name, processor, unimportance_orders, prompt, image_url)
```

### 2. Saving Results
The script saves inference results in a CSV file (`results.csv`) with two columns:
- `num_of_layer`: The number of layers remaining after pruning.
- `response`: The model’s response to the prompt.

## Clearing GPU Memory
After each inference, the script removes the pruned model from the GPU and clears cache:
```python
del model_pruned
torch.cuda.empty_cache()
gc.collect()
```

## Contributions
Feel free to submit pull requests for improvements or additional features!

## License
This project is open-source under the MIT License.

