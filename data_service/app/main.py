from fastapi import FastAPI

app = FastAPI(title="Data Service - SP500 Volatility")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# TODO: Implement endpoint /features/latest
# TODO: Implement endpoint /features/history
