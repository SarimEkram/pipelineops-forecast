from fastapi import FastAPI

app = FastAPI(title="PipelineOps API")

@app.get("/health")
def health():
    return {"status": "ok"}
