import streamlit as st
import pandas as pd
from crewai import Agent, Task, Crew, Process
from langchain.llms import OpenAI
from crewai_tools import tool
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import io
import os

# Assuming GPT-4.0 mini is accessible via OpenAI's API
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

# Set up API keys (you should use environment variables in a real application)
os.environ["OPENAI_API_KEY"] = "Your OpenAI Key"

@tool('analyze_data')
def analyze_data(data: str):
    """Analyze the given data sample and provide insights"""
    # This is a placeholder. In a real scenario, you'd implement more sophisticated analysis here.
    return f"Analysis of data: {data[:500]}..."  # Truncated for brevity

@tool('validate_data')
def validate_data(data: str):
    """Validate the correctness and consistency of the given data sample"""
    # This is a placeholder. In a real scenario, you'd implement actual validation logic here.
    return f"Validation of data: {data[:500]}... Data appears to be consistent."  # Truncated for brevity

# Define our agents
data_analyst = Agent(
    role='Data Analyst',
    goal='Analyze structural data from files and determine its meaning',
    backstory="You're an expert in interpreting complex structural data from various file formats.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[analyze_data]
)

data_validator = Agent(
    role='Data Validator',
    goal='Verify the correctness and integrity of structural data from files',
    backstory="You're meticulous about data quality and accuracy in various file formats.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[validate_data]
)

def analyze_file(file, file_type):
    # Read the file based on its type
    if file_type == 'csv':
        df = pd.read_csv(file)
    else:  # Excel files
        df = pd.read_excel(file)
    
    # Convert dataframe to string for analysis
    data_str = df.to_string()
    
    # Create a vector store from the data
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts([data_str], embeddings)
    
    # Define tasks
    analysis_task = Task(
        description=f"Analyze the structural data from the {file_type.upper()} file. Provide insights on its structure and potential purpose.",
        agent=data_analyst,
        expected_output='A detailed analysis of the data structure and its potential purpose'
    )

    validation_task = Task(
        description=f"Verify if the structural data from the {file_type.upper()} file is correct and consistent. Check for inconsistencies or errors in the data.",
        agent=data_validator,
        expected_output='A validation report highlighting any inconsistencies or errors in the data'
    )

    # Create and run the crew
    crew = Crew(
        agents=[data_analyst, data_validator],
        tasks=[analysis_task, validation_task],
        process=Process.sequential,
        verbose=2,
        full_output=True
    )

    result = crew.kickoff()
    
    return result

def main():
    st.title("Structural Data Analysis App")

    uploaded_files = st.file_uploader("Choose Excel or CSV files", accept_multiple_files=True, type=['xlsx', 'xls', 'csv'])

    if st.button("Analyze Files"):
        if uploaded_files:
            for file in uploaded_files:
                file_type = file.name.split('.')[-1].lower()
                st.write(f"Analyzing {file.name}...")
                result = analyze_file(file, file_type)
                st.subheader(f"Analysis Result for {file.name}:")
                st.write(result.raw)  # Display raw output
                st.write("Token Usage:")
                st.write(result.token_usage)
                for task_output in result.tasks_output:
                    st.write(f"Task: {task_output.task_description}")
                    st.write(f"Output: {task_output.output}")
        else:
            st.warning("Please upload at least one Excel or CSV file.")

if __name__ == "__main__":
    main()