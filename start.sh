export $(cat .env)
uvicorn main:app --reload --port 8001