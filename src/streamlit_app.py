import streamlit as st
import json
from service import process_financial_question, process_json_input

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
                st.session_state.df, st.session_state.current_data = process_json_input(json_input)
                st.success("JSON data processed successfully!")
                
                # Display data preview
                st.subheader("Data Preview")
                st.json(st.session_state.current_data.get('table', {}))
            except ValueError as e:
                st.error(str(e))
        
        # Additional context
        context = st.text_area("Additional Context (Optional)", height=100)
    
    # Main content area
    if st.session_state.df is not None and not st.session_state.df.empty:
        # Question input
        question = st.text_input("Enter your financial question:")
        
        if question:
            with st.spinner(f"Analyzing using {st.session_state.client_choice}..."):
                try:
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
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
        
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