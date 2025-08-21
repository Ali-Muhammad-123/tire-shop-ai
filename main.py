# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Any
import json
import re
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


# -------- Setup --------
# Safely read your API key from environment variables

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Reference Finder API", version="1.0.0")

ASSISTANT_ID = "asst_3CF1iUqL11WbPZe1F805LhLm"


class QueryRequest(BaseModel):
    query: str = Field(
        ..., description="Natural-language query to search references for"
    )


class QueryResponse(BaseModel):
    result: Any


def clean_to_json(raw_text: str):
    """
    Cleans a raw string containing JSON-like content and converts it into a Python object (list or dict).

    Args:
        raw_text (str): The raw text that looks like JSON but may contain extra characters or escape sequences.

    Returns:
        list | dict: Parsed JSON object.

    Raises:
        ValueError: If the string cannot be parsed into valid JSON.
    """
    try:
        # 1️⃣ Remove unwanted whitespace and line breaks
        cleaned = raw_text.strip()

        # 2️⃣ Remove escape characters like \n and extra backslashes
        cleaned = re.sub(r"\\n", "", cleaned)
        cleaned = re.sub(r"\\", "", cleaned)

        # 3️⃣ Parse into JSON
        return json.loads(cleaned)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON structure. Details: {e}")


@app.post("/find", response_model=QueryResponse)
def find_reference(req: QueryRequest):
    try:
        response = client.responses.create(
            model="gpt-5-nano",
            prompt={
                "id": "pmpt_68a79bb2a2748193ae50edb75dc791fc0271289fe1eedd4d",
                "version": "5",
            },
            input=req.query,
        )

        # Safely extract the text output
        result_text = getattr(response, "output_text", None)
        if not result_text:
            result_text = ""
            if getattr(response, "output", None):
                for part in response.output:
                    for c in getattr(part, "content", []) or []:
                        if c.get("type") == "output_text" and "text" in c:
                            result_text += c["text"]

        if not result_text:
            result_text = "No textual result returned."

        try:
            # Attempt to parse the result as JSON
            result = clean_to_json(result_text)
        except ValueError:
            # If parsing fails, return the raw text
            result = result_text
        return QueryResponse(result=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------- Local dev entrypoint --------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
