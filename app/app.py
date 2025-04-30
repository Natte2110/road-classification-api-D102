from fastapi import FastAPI

app = FastAPI(
    title="Road Surface Classification API",
    version="1.0",
    description="""
    This API provides endpoints for managing machine learning models, performing inference,
    and monitoring job execution. The API supports model uploads, batch inference, 
    and real-time model comparisons.
    
    **Features:**
    - List and manage models
    - Perform inference with uploaded images
    - Monitor execution jobs
    - Retrieve model performance metrics
    """,
    contact={
        "name": "UDTIP API Support",
        "url": "https://ogc.org/contacts",
        "email": "support@ogc.org"
    },
    license_info={
        "name": "OGC License",
        "url": "http://www.ogc.org/legal/"
    },
    openapi_url="/api/v1/openapi.json"
)