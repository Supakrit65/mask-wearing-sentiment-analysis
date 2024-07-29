import re
import os
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import pandas as pd
from collections import Counter
from datasets import load_dataset

# Specify the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load the Hugging Face model and tokenizer
model_name = "your-model-name"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

# Ensure the model is in inference mode
model.eval()

def generate_prediction(instruction, input_text):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction:
    {}
    
    ### Input:
    {}
    
    ### Response:
    {}"""

    inputs = tokenizer(
        [alpaca_prompt.format(instruction, input_text, "")], return_tensors="pt", padding=True, truncation=True
    ).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    
    prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    response_text = prediction[0].split("### Response:")[-1].strip()
    
    try:
        stance_word = re.match(r'\w+', response_text).group(0)
    except Exception as e:
        print(f"Error occurred during stance word extraction: {e}")
        print(f"Full response text: {response_text}")
        print(f"Prediction: {prediction}")
        stance_word = None
    
    return stance_word

def unique(list1):
    unique_list = list(set(list1))
    for x in unique_list:
        print(x)
    value_counts = Counter(list1)
    print("\nValue counts for class labels:")
    for label, count in value_counts.items():
        print(f"{label}: {count}")

def preprocess_predictions(predictions):
    processed_predictions = []
    for prediction in predictions:
        prediction_lower = prediction.lower()
        if 'favor' in prediction_lower or 'favour' in prediction_lower:
            processed_predictions.append('Favorable')
        elif 'against' in prediction_lower:
            processed_predictions.append('Against')
        elif 'neutral' in prediction_lower:
            processed_predictions.append('Neutral')
        else:
            print(prediction)
            raise ValueError('Unexpected prediction value')
    return processed_predictions

us_comments_predictions = []
model.config.pad_token_id = tokenizer.pad_token_id

# Load dataset from Hugging Face datasets library
dataset = load_dataset('your-dataset-name', split='train')
to_label_dataset = pd.DataFrame(dataset)
print(to_label_dataset.shape)

for index, example in tqdm(to_label_dataset.iterrows(), desc="Processing examples", unit="example"):
    instruction = example["instruction"]
    input_text = example["input"]
    predicted_output = generate_prediction(instruction, input_text)
    us_comments_predictions.append(predicted_output)

predictions = preprocess_predictions(us_comments_predictions)
print('**Predicts**')
unique(predictions)
print('-' * 50)

to_label_dataset['pred_stance'] = predictions
print(to_label_dataset.shape)

to_label_dataset.to_csv('english_mask_done_v3.csv', index=False)

print("Stance Prediction Done!")
