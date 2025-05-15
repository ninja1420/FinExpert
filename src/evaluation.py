import json
import pandas as pd
import os
from dotenv import load_dotenv
import groq
import openai
from typing import Dict, List, Union
import numpy as np
from tqdm import tqdm

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

def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    # Remove whitespace, convert to lowercase
    answer = answer.strip().lower()
    
    # Remove common units and symbols
    answer = answer.replace('$', '').replace('%', '').replace(',', '')
    answer = answer.replace('million', '').replace('billion', '').replace('thousand', '')
    answer = answer.strip()  # Remove any remaining whitespace
    
    try:
        # Try to convert to float and round to nearest integer
        return str(round(float(answer)))
    except:
        # If not a number, return normalized string
        return answer

def compare_answers(predicted: str, actual: str) -> bool:
    """Compare predicted and actual answers."""
    pred_norm = normalize_answer(predicted)
    actual_norm = normalize_answer(actual)
    
    try:
        # Try numerical comparison if both are numbers
        pred_float = float(pred_norm)
        actual_float = float(actual_norm)
        # Compare rounded integers
        return pred_float == actual_float
    except:
        # Fall back to string comparison
        return pred_norm == actual_norm

def process_question(client: Union[groq.Client, openai.Client], model: str, 
                    question: str, table_data: Dict, context: str = "") -> str:
    """Process a single question using the selected API."""
    try:
        # Prepare prompt with context
        prompt = f"""
        You are a financial assistant tasked with answering quantitative questions based on financial documents. 

        Below is the financial context, a specific question, structured table data, and pre-parsed numeric field values.

        Your job is to extract the correct numerical answer to the question by following these structured steps internally. 
        **You should only return the final answer — no explanation or reasoning is required.**

        ---

        **Financial Context**:
        {context}

        **Question**:
        {question}

        **Table Data**:
        {json.dumps(table_data, indent=2)}

        ---

        **Instructions to Solve Internally**:

        1. Carefully read and understand the financial question.
        2. Identify the relevant numeric fields from the table.
        3. Perform any required calculations such as:
        - Year-over-year difference: `value_year_2 - value_year_1`
        - Percentage change: `(value_year_2 - value_year_1) / value_year_1 * 100`
        - Ratios or additions/subtractions across fields.
        4. Return only the **final numeric answer** (e.g., `14.1%`, `123 million`, `-56.2`).

        Do NOT return:
        - Reasoning
        - Step-by-step text
        - Any justification
        - Any explanation or code

        Just return the final answer on a single line.
        """

        # Call API
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analysis expert. Provide answer of given question based on the provided financial data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return ""

def get_qa_pairs(item: Dict) -> List[Dict]:
    """Extract all QA pairs from an item, handling both single and multiple question cases."""
    qa_pairs = []
    
    # Check for single qa
    if 'qa' in item:
        qa_pairs.append(item['qa'])
    
    # Check for multiple qa_X keys
    qa_keys = [k for k in item.keys() if k.startswith('qa_')]
    for key in sorted(qa_keys):  # Sort to process in order qa_0, qa_1, etc.
        qa_pairs.append(item[key])
    
    return qa_pairs

def evaluate_model(dev_json_path: str, client_choice: str = "Groq") -> Dict:
    """Evaluate model performance on dev.json dataset."""
    # Load dev.json
    with open(dev_json_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    print(len(dev_data))
    
    # Initialize client
    client = get_client(client_choice)
    model = get_model_name(client_choice)
    
    results = {
        'total_questions': 0,  # Will be calculated as we process
        'processed_questions': 0,
        'correct_answers': 0,
        'incorrect_answers': [],
        'errors': []
    }
    
    # Process each item
    for idx, item in enumerate(tqdm(dev_data, desc="Processing items")):
        try:
            # Build context from pre_text and post_text
            context = f"{item.get('pre_text', '')} {item.get('post_text', '')}"
            
            # Get table data (prefer original if available)
            table_data = item.get('table_ori', item.get('table', {}))
            
            # Get all QA pairs for this item
            qa_pairs = get_qa_pairs(item)
            results['total_questions'] += len(qa_pairs)
            
            # Process each question in this item
            for qa in qa_pairs:
                try:
                    question = qa['question']
                    actual_answer = qa['answer']
                    
                    # Get model's prediction
                    predicted_answer = process_question(client, model, question, table_data, context)
                    
                    # Only count questions that were successfully processed
                    if predicted_answer:  # Skip empty responses
                        results['processed_questions'] += 1
                        
                        # Compare answers
                        is_correct = compare_answers(predicted_answer, actual_answer)
                        
                        if is_correct:
                            results['correct_answers'] += 1
                        else:
                            results['incorrect_answers'].append({
                                'question': question,
                                'predicted': predicted_answer,
                                'actual': actual_answer
                            })
                            
                except Exception as e:
                    results['errors'].append({
                        'index': f"{idx}_qa",
                        'error': str(e)
                    })
                    
        except Exception as e:
            results['errors'].append({
                'index': idx,
                'error': str(e)
            })
    
    # Calculate accuracy based only on processed questions
    if results['processed_questions'] > 0:
        results['accuracy'] = results['correct_answers'] / results['processed_questions']
    else:
        results['accuracy'] = 0.0
    
    # Add error rate information
    results['error_rate'] = len(results['errors']) / results['total_questions']
    results['successful_processing_rate'] = results['processed_questions'] / results['total_questions']
    
    return results

def main():
    # Path to dev.json
    dev_json_path = "Data/data/dev.json"
    
    models = ["OpenAI"]
    all_results = {}
    
    for model in models:
        print(f"\nEvaluating {model} model...")
        results = evaluate_model(dev_json_path, model)
        all_results[model] = results
        
        # Print results
        print(f"\n{model} Results:")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Successfully Processed: {results['processed_questions']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Accuracy (of processed): {results['accuracy']:.2%}")
        print(f"Processing Success Rate: {results['successful_processing_rate']:.2%}")
        print(f"Error Rate: {results['error_rate']:.2%}")
        print(f"Number of Errors: {len(results['errors'])}")
        
        # Print some incorrect examples
        print("\nSample Incorrect Predictions:")
        for i, incorrect in enumerate(results['incorrect_answers'][:5]):  # Show first 5 incorrect
            print(f"\nQuestion {i+1}: {incorrect['question']}")
            print(f"Predicted: {incorrect['predicted']}")
            print(f"Actual: {incorrect['actual']}")
    
    # Save results to file
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main() 