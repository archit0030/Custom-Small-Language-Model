import torch
from transformers import BartTokenizer, T5Tokenizer
from peft import PeftModel
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration
import streamlit as st

# 1. Load model and tokenizer
model_path = './finetuned_robotic_t5'
tokenizer = T5Tokenizer.from_pretrained(model_path)
base_model = T5ForConditionalGeneration.from_pretrained('./finetuned_robotic_t5')
model = PeftModel.from_pretrained(base_model, model_path)
model = model.merge_and_unload()  # Merge LoRA adapters

# 2. Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 3. Prediction function
def predict_actions(instruction):
    # Tokenize input
    inputs = tokenizer(
        instruction,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(device)
    
    # Generate actions
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=64,
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decoded = decoded.lower()  # Force lowercase
    return [action.strip() for action in decoded.split() if action.strip()]

# Streamlit interface
st.title("Robotic Action Predictor")

# Input text box
instruction = st.text_input("Enter your instruction:", "")

# Predict button
if st.button("Predict Actions"):
    if instruction:
        try:
            actions = predict_actions(instruction)
            st.subheader("Predicted Actions:")
            for i, action in enumerate(actions, 1):
                st.write(f"{i}. {action}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a valid instruction")