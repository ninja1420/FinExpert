from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from service import process_financial_question
from financial_utils import process_json_data
import uvicorn


app = FastAPI(
    title="Financial Reasoning Assistant",
    description="A financial reasoning assistant that uses LLMs to answer questions about financial data.",
    version="0.1.0",
)


class FinancialQuestion(BaseModel):
    question: str
    json_data: str
    context: Optional[str] = None
    client_choice: Optional[str] = 'Groq'

class FinancialAnswer(BaseModel):
    answer: str

@app.post("/answer", response_model=FinancialAnswer)
async def analyse_financial_data(question: FinancialQuestion):
    try:
        # Process the JSON data
        df = process_json_data(question.json_data)

        # Process the question
        answer = process_financial_question(question.question, df, question.context, question.client_choice)

        return FinancialAnswer(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    


