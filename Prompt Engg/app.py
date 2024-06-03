import pandas as pd
import streamlit as st
from groq import Groq
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Load dataset
df = pd.read_csv("data/Copy of AI Project Data Training Sheet - Data Sheet.csv")

# Preprocess dataset
df.columns = df.columns.str.strip()  # Ensure no leading/trailing spaces in column names
input_prompts = df['Hook'] + ' ' + df['Build Up']
target_outputs = df['Body'] + ' ' + df['CTA']

# Select few-shot prompts
few_shot_prompts = list(zip(input_prompts[:5], target_outputs[:5]))  # Select first 5 examples for simplicity

# Initialize Groq client
client = Groq(api_key="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")


# Placeholder function for calculating loss
def calculate_loss(generated_output, target):
    # Example loss calculation using edit distance
    return sum(1 for a, b in zip(generated_output, target) if a != b)

# Placeholder function for updating model parameters
def update_model_parameters(loss):
    # This is a placeholder; actual model parameter update is complex
    pass

# Function for evaluating model performance
def evaluate_model_performance(predictions, targets):
    # Calculate evaluation metrics
    true_positives = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    false_positives = len(predictions) - true_positives
    precision = precision_score(targets, predictions, average='macro')
    recall = recall_score(targets, predictions, average='macro')
    f1 = f1_score(targets, predictions, average='macro')
    accuracy = accuracy_score(targets, predictions)
    
    evaluation_metrics = {
        "F1 score": f1,
        "True positives": true_positives,
        "False positives": false_positives,
        "Recall": recall,
        "Accuracy": accuracy
    }
    return evaluation_metrics

# Define fine-tuning function with retry mechanism
def fine_tune_model(client, few_shot_prompts, num_epochs):
    model_id = "llama3-70b-8192"
    all_predictions = []
    all_targets = []
    
    for epoch in range(num_epochs):
        for prompt, target in few_shot_prompts:
            success = False
            attempts = 0
            while not success and attempts < 5:  # Retry up to 5 times
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_id
                    )
                    generated_output = chat_completion.choices[0].message.content
                    all_predictions.append(generated_output)
                    all_targets.append(target)
                    
                    loss = calculate_loss(generated_output, target)
                    update_model_parameters(loss)
                    
                    success = True
                except Exception as e:
                    if 'rate limit' in str(e):
                        attempts += 1
                        wait_time = 2 ** attempts  # Exponential backoff
                        print(f"Rate limit reached, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Unexpected error: {e}")
                        success = True  # Exit loop if an unexpected error occurs

    return evaluate_model_performance(all_predictions, all_targets)

# Fine-tune the model
num_epochs = 5
evaluation_metrics = fine_tune_model(client, few_shot_prompts, num_epochs)

# Set up Streamlit app
st.title('AI Model Prompt Engineering and Evaluation')
st.write('This app allows you to explore and evaluate AI model prompts using a provided dataset.')

# Display the dataset
st.subheader('Dataset')
st.write(df)


# Prompt input
st.subheader('Create a Prompt')
prompt_text = st.text_area('Enter your prompt here:')

# Generate response
if st.button('Generate'):
    st.subheader('Generated Response')
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_text}],
            model="llama3-70b-8192"
        ).choices[0].message.content
        st.write(response)
    except Exception as e:
        st.write(f"Error: {e}")

# Evaluation metrics
st.subheader('Evaluate the Prompt')
rating = st.slider('Rate the quality of the generated prompt:', 0, 5, 3)
feedback = st.text_area('Provide feedback on the generated prompt:')

# Save feedback
if st.button('Submit Feedback'):
    st.write('Thank you for your feedback!')

# Display evaluation metrics
st.subheader('Evaluation Metrics')
st.write(evaluation_metrics)
