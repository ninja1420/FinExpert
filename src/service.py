import os
import groq
import openai
import json
import pandas as pd
from typing import Dict, Union
from dotenv import load_dotenv
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
            raise ValueError("Groq API key not found in environment variables!")
        return groq.Client(api_key=groq_api_key)
    else:  # OpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables!")
        return openai.Client(api_key=openai_api_key)

def get_model_name(client_choice: str) -> str:
    """Get the appropriate model name based on client choice."""
    if client_choice == "Groq":
        return "llama-3.3-70b-versatile"
    else:  # OpenAI
        return "gpt-4-turbo-preview"

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

def process_json_input(json_input: str) -> tuple:
    """Process JSON input and return DataFrame and original data."""
    try:
        json_data = json.loads(json_input)
        print(json_data)
        df = process_json_data(json_data)
        if df.empty:
            raise ValueError("No valid table data found in the JSON input")
        return df, json_data
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format. Please check your input.")
    except Exception as e:
        raise ValueError(f"Error processing JSON data: {str(e)}")
