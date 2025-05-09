import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import groq
import openai
import json
from typing import Dict, Union
from financial_utils import (
    calculate_percentage_change,
    extract_financial_values,
    parse_financial_table,
    format_currency,
    format_percentage,
    load_financial_json,
    process_json_data,
    analyze_financial_data
)

# Load environment variables
load_dotenv()

def get_client(client_choice: str) -> Union[groq.Client, openai.Client]:
    """Get the appropriate API client based on user choice."""
    if client_choice == "Groq":
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("Groq API key not found in environment variables!")
            st.stop()
        return groq.Client(api_key=groq_api_key)
    else:  # OpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("OpenAI API key not found in environment variables!")
            st.stop()
        return openai.Client(api_key=openai_api_key)

def get_model_name(client_choice: str) -> str:
    """Get the appropriate model name based on client choice."""
    if client_choice == "Groq":
        return "llama-3.3-70b-versatile"
    else:  # OpenAI
        return "gpt-4-turbo-preview"

def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'client_choice' not in st.session_state:
        st.session_state.client_choice = "Groq"

def process_financial_question(question: str, df: pd.DataFrame, context: str = "", client_choice: str = "Groq") -> str:
    """Process financial question using selected API."""
    try:
        # Get appropriate client and model
        client = get_client(client_choice)
        model = get_model_name(client_choice)
        
        # Analyze the financial data
        analysis_result = analyze_financial_data(df, question)
        
        # Prepare the prompt
        prompt = f"""
        You are a financial assistant tasked with answering quantitative questions based on financial documents. 

        Below is the financial context, a specific question, structured table data, and pre-parsed numeric field values.

        Your job is to extract the correct numerical answer to the question by following these structured steps internally. 
        **You should only return the final answer — no explanation or reasoning is required.**

        ---

        📄 **Financial Context**:
        {context}

        ❓ **Question**:
        {question}

        📊 **Table Data**:
        {json.dumps(analysis_result['table_data'], indent=2)}

        🔢 **Available Numeric Fields**:
        {', '.join(analysis_result['numeric_fields'])}

        📌 **Pre-calculated Field Values**:
        {json.dumps(analysis_result['calculations'], indent=2)}

        ---

        🧮 **Instructions to Solve Internally**:

        1. Carefully read and understand the financial question.
        2. Identify the relevant numeric fields from the table or pre-parsed values.
        3. Perform any required calculations such as:
        - Year-over-year difference: `value_year_2 - value_year_1`
        - Percentage change: `(value_year_2 - value_year_1) / value_year_1 * 100`
        - Ratios or additions/subtractions across fields.
        4. Always prefer using values from `Field Values` section if already provided.
        5. Return only the **final numeric answer** (e.g., `14.1%`, `123 million`, `-56.2`).

        🚫 Do NOT return:
        - Reasoning
        - Step-by-step text
        - Any justification
        - Any explanation or code

        ✅ Just return the final answer on a single line.

        """

        
        # Call appropriate API
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analysis expert. Provide answer of given question based on the provided financial data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        # Extract response based on client type
        if client_choice == "Groq":
            return completion.choices[0].message.content
        else:  # OpenAI
            return completion.choices[0].message.content
            
    except Exception as e:
        return f"Error processing question: {str(e)}"

def main():
    st.set_page_config(page_title="Financial Reasoning Assistant", layout="wide")
    st.title("Financial Reasoning Assistant 💰")
    
    initialize_session_state()
    
    # Sidebar for settings and data input
    with st.sidebar:
        st.header("Settings")
        
        # API Selection
        st.session_state.client_choice = st.radio(
            "Select API Provider",
            ["Groq", "OpenAI"],
            index=0 if st.session_state.client_choice == "Groq" else 1,
            help="Choose which API to use for analysis"
        )
        
        st.header("Input Financial Data")
        
        # JSON input text area
        json_input = st.text_area("Enter JSON Data", height=300, 
                                help="Paste a single JSON record containing financial data")
        
        if json_input:
            try:
                json_data = json.loads(json_input)
                st.session_state.df = process_json_data(json_data)
                if not st.session_state.df.empty:
                    st.session_state.current_data = json_data
                    st.success("JSON data processed successfully!")
                    
                    # Display data preview
                    st.subheader("Data Preview")
                    st.json(json_data.get('table', {}))
                else:
                    st.error("No valid table data found in the JSON input")
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check your input.")
            except Exception as e:
                st.error(f"Error processing JSON data: {str(e)}")
        
        # Additional context
        context = st.text_area("Additional Context (Optional)", height=100)
    
    # Main content area
    if st.session_state.df is not None and not st.session_state.df.empty:
        # Question input
        question = st.text_input("Enter your financial question:")
        
        if question:
            with st.spinner(f"Analyzing using {st.session_state.client_choice}..."):
                response = process_financial_question(
                    question,
                    st.session_state.df,
                    context,
                    st.session_state.client_choice
                )
                
                # Display response
                st.subheader("Analysis")
                st.write(response)
                
                # Update chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": response,
                    "data": st.session_state.current_data.get('table', {}),
                    "api_used": st.session_state.client_choice
                })
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Previous Questions")
            for i, qa in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q: {qa['question']} (via {qa['api_used']})"):
                    st.write("Data analyzed:")
                    st.json(qa['data'])
                    st.write("Analysis:")
                    st.write(qa['answer'])
    else:
        st.info("Please enter JSON data in the sidebar to begin.")

if __name__ == "__main__":
    main() 