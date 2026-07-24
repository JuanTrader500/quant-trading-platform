from fastapi import FastAPI

app = FastAPI(title="ML Service - SP500 Volatility")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# TODO: Implement endpoint /predict
# TODO: Implement endpoint /retrain
# TODO: Implement endpoint /metrics
