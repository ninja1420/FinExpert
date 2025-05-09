import pandas as pd
import numpy as np
import json
from typing import Dict, List, Union, Optional

def calculate_percentage_change(current: float, previous: float) -> float:
    """Calculate percentage change between two values."""
    if previous == 0:
        return float('inf') if current > 0 else float('-inf')
    return ((current - previous) / abs(previous)) * 100

def extract_financial_values(data: Union[pd.DataFrame, Dict], keys: List[str]) -> Dict[str, float]:
    """Extract specific financial values from data structure."""
    if isinstance(data, pd.DataFrame):
        return {key: data[key].iloc[-1] for key in keys if key in data.columns}
    elif isinstance(data, dict):
        return {key: data[key] for key in keys if key in data}
    return {}

def parse_financial_table(table_text: str) -> pd.DataFrame:
    """Parse financial table text into pandas DataFrame."""
    # Split the text into lines and clean them
    lines = [line.strip() for line in table_text.split('\n') if line.strip()]
    
    # Extract headers and data
    headers = lines[0].split()
    data = []
    for line in lines[1:]:
        values = line.split()
        if len(values) == len(headers):
            data.append(values)
    
    return pd.DataFrame(data, columns=headers)

def format_currency(value: float) -> str:
    """Format number as currency string."""
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format number as percentage string."""
    return f"{value:.2f}%"

def load_financial_json(file_path: str) -> Dict:
    """Load and parse financial data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_json_data(json_data: Dict) -> pd.DataFrame:
    """Process a single JSON record into a pandas DataFrame for analysis.
    Extracts only the table data, ignoring questions and answers."""
    try:
        # Extract only the table data from the JSON
        if isinstance(json_data, dict):
            table_data = json_data.get('table', {})
            if table_data:
                # Convert the table data to DataFrame
                df = pd.DataFrame([table_data])
                
                # Convert numeric columns to appropriate types
                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except (ValueError, TypeError):
                        continue
                
                return df
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing JSON data: {str(e)}")
        return pd.DataFrame()

def analyze_financial_data(df: pd.DataFrame, question: str) -> Dict[str, Union[str, float, Dict]]:
    """Analyze financial data based on the question asked.
    Returns a dictionary containing analysis results."""
    try:
        # Convert all numeric columns to float
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].astype(float)
        
        result = {
            'question': question,
            'table_data': df.to_dict('records')[0],  # Original table data
            'numeric_fields': list(numeric_cols),
            'calculations': {}
        }
        
        # Calculate basic statistics for numeric fields
        for col in numeric_cols:
            result['calculations'][col] = {
                'value': df[col].iloc[0],
                'type': 'numeric'
            }
        
        return result
    except Exception as e:
        print(f"Error analyzing data: {str(e)}")
        return {
            'question': question,
            'error': str(e)
        } 