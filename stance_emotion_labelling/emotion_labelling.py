import pandas as pd
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
from enum import Enum

class Emotions(Enum):
    ANGER = "anger"
    FEAR = "fear"
    JOY = "joy"

def get_emotional_intensity(df, target_emotion: Emotions, emo_pipeline):
    input_texts = df['english_comment_text']
    predictions = []
    for comment_text in tqdm(input_texts, desc=f"Processing {target_emotion.value} intensity"):
        prompt = f"""
        Human: 
        Task: Assign a numerical value between 0 (least {target_emotion.value}) and 1 (most {target_emotion.value}) to represent the intensity of emotion {target_emotion.value} expressed in the text.
        Text: {comment_text}
        Emotion: {target_emotion.value}
        Intensity Score:"""
        prediction = emo_pipeline(prompt, max_length=2048)[0]['generated_text']
        predictions.append(prediction.split("Intensity Score:")[-1].strip())
    return predictions

def get_emotional_classification(df, emo_pipeline):
    input_texts = df['english_comment_text']
    predictions = []
    for comment_text in input_texts:
        prompt = f"""
        Human: 
        Task: Categorize the text's emotional tone as either 'neutral or no emotion' or identify the presence of one or more of the specified emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust).
        Text: {comment_text}
        This text contains emotions:"""
        prediction = emo_pipeline(prompt)[0]['generated_text']
        predictions.append(prediction.split("This text contains emotions:")[-1].strip())
    return predictions

pipe = pipeline("text-generation", model="lzw1008/Emollama-chat-13b")

to_label_dataset = load_dataset("hf/your-dataset-name")
to_label_dataset = pd.DataFrame(to_label_dataset)

for emotion in Emotions:
    to_label_dataset[f"{emotion.value}_intensity"] = get_emotional_intensity(to_label_dataset, emotion, pipe)

to_label_dataset['emotional_classification'] = get_emotional_classification(to_label_dataset, pipe)

output_file = "emotion_analysis_th_mask_train.csv"
to_label_dataset.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
