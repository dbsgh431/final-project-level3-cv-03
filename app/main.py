from fastapi import FastAPI
from routers import Det
import uvicorn

app = FastAPI()

if __name__ == '__main__':
    app.include_router(Det.router)
    uvicorn.run(app, host='0.0.0.0', port=30011)