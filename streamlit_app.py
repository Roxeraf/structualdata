import streamlit as st
import pandas as pd
import openai
import os

# Set up OpenAI API key
openai.api_key = "Your-OpenAI-API-Key"  # Replace with your actual API key

def analyze_data(data: str):
    """Analyze the given data sample and provide insights"""
    prompt = f"Analyze the following data and provide insights on its structure and potential purpose:\n\n{data}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a data analyst expert in interpreting complex structural data."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

def validate_data(data: str):
    """Validate the correctness and consistency of the given data sample"""
    prompt = f"Validate the following data for correctness and consistency. Highlight any inconsistencies or errors:\n\n{data}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a data validator expert in checking data quality and accuracy."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

def analyze_file(file, file_type):
    # Read the file based on its type
    if file_type == 'csv':
        df = pd.read_csv(file)
    else:  # Excel files
        df = pd.read_excel(file)
    
    # Convert dataframe to string for analysis
    data_str = df.head(10).to_string()  # Analyze first 10 rows
    
    # Perform analysis and validation
    analysis_result = analyze_data(data_str)
    validation_result = validate_data(data_str)
    
    return analysis_result, validation_result

def main():
    st.title("Structural Data Analysis App")

    uploaded_files = st.file_uploader("Choose Excel or CSV files", accept_multiple_files=True, type=['xlsx', 'xls', 'csv'])

    if st.button("Analyze Files"):
        if uploaded_files:
            for file in uploaded_files:
                file_type = file.name.split('.')[-1].lower()
                st.write(f"Analyzing {file.name}...")
                analysis_result, validation_result = analyze_file(file, file_type)
                st.subheader(f"Analysis Result for {file.name}:")
                st.write(analysis_result)
                st.subheader(f"Validation Result for {file.name}:")
                st.write(validation_result)
        else:
            st.warning("Please upload at least one Excel or CSV file.")

if __name__ == "__main__":
    main()