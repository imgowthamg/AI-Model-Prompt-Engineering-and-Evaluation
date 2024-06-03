# AI Model Prompt Engineering and Evaluation

This Streamlit app provides a user interface to explore and evaluate AI model prompts using a provided dataset. The app interacts with the GROQ API for fine-tuning a language model and generating responses based on user prompts.

## Code Explanation:

### 1. Dataset Loading and Preprocessing:
- The app loads the dataset from a CSV file using pandas.
- The dataset is preprocessed to ensure no leading or trailing spaces in column names.
- Input prompts and target outputs are concatenated from specific columns in the dataset.

### 2. Fine-tuning Model:
- The code fine-tunes a language model using the GROQ API.
- It initializes a GROQ client with the appropriate API key.
- A placeholder function is used to calculate loss and update model parameters.
- The model is fine-tuned using few-shot prompts from the dataset.

### 3. Streamlit App Setup:
- The Streamlit app is set up to provide a user interface for interacting with the model.
- It displays a title and description of the app.

### 4. Display Dataset:
- The dataset is displayed using Streamlit's `st.write()` function.

### 5. Prompt Input and Response Generation:
- Users can input prompts using a text area.
- Upon clicking the "Generate" button, the app sends the prompt to the fine-tuned model via the GROQ API.
- The generated response is displayed in the app.

### 6. Prompt Evaluation and Feedback:
- Users can rate the quality of the generated prompt using a slider.
- They can also provide feedback on the generated prompt using a text area.
- Feedback can be submitted by clicking the "Submit Feedback" button.

### 7. Display Evaluation Metrics:
- Evaluation metrics such as F1 score, precision, recall, and accuracy are displayed at the end of the app.

## Dataset Splitting and Workflow:
- The dataset is split into input prompts and target outputs.
- Few-shot prompts, consisting of the first 5 examples, are selected for simplicity.
- These prompts are used for fine-tuning the model.
- The fine-tuned model is then used to generate responses based on user prompts entered via the Streamlit app.
- Users can evaluate the quality of the generated prompts and provide feedback.




![Screenshot 2024-06-03 153140](https://github.com/imgowthamg/AI-Model-Prompt-Engineering-and-Evaluation/assets/119653141/f88282b6-01c5-4ab3-ad85-fe89cdb30768)
![Screenshot 2024-06-03 153327](https://github.com/imgowthamg/AI-Model-Prompt-Engineering-and-Evaluation/assets/119653141/be6c7635-846f-450f-91a0-7ca6dc82da38)
![Screenshot 2024-06-03 153353](https://github.com/imgowthamg/AI-Model-Prompt-Engineering-and-Evaluation/assets/119653141/20048fd7-a4e4-41ae-af0a-550ccbf8248a)
