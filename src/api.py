import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd
from service import process_financial_question, process_json_input
os.makedirs('logs', exist_ok=True)

    # Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/financial_analysis.log'),
        logging.StreamHandler()  # This will also log to console
    ]
)

logger = logging.getLogger(__name__)
logger.info("Starting FastAPI application")

app = FastAPI(
    title="Financial Analysis API",
    description="API for analyzing financial data using LLMs",
    version="1.0.0",
    debug=True  # Enable debug mode
)

class FinancialQuestion(BaseModel):
    question: str
    json_data: str
    context: Optional[str] = ""
    client_choice: Optional[str] = "Groq"

class AnalysisResponse(BaseModel):
    answer: str
    error: Optional[str] = None

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_financial_data(request: FinancialQuestion):
    """
    Analyze financial data and answer questions using LLMs.
    
    Args:
        request (FinancialQuestion): The question and data to analyze
        
    Returns:
        AnalysisResponse: The analysis result or error message
    """
    try:
        # Process the JSON input
        logger.info(f"Received request: {request}")
        df, _ = process_json_input(request.json_data)
        logger.info(f"Processed DataFrame: {df.to_dict()}")

        
        # Process the question
        answer = process_financial_question(
            question=request.question,
            df=df,
            context=request.context,
            client_choice=request.client_choice
        )
        logger.info(f"Generated answer: {answer}")
        
        return AnalysisResponse(answer=answer)
        
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint called")
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
   
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload
        log_level="info",  # Set log level to debug
        workers=1  # Use single worker for debugging
    ) 