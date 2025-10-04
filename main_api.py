# main_api.py
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import pandas as pd
import json
import logging
from datetime import datetime
import requests
import boto3
from config import *
import sys

from medical_utils import fetch_lab_data, load_biomarker_master_table, integrate_data, DataIntegrationError
from medical_utils import write_excel_to_s3, get_aws_credentials_from_secrets_manager

# Import your LangGraph agent graph
from LLM_main import agent_graph

# ---------------------------------
# Logging Config (CloudWatch only)
# ---------------------------------
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG if you want more verbose logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]  # log to stdout only
)

logger = logging.getLogger(__name__)
####  

# Fetch env vars from App Runner
secret_name = os.getenv("AWS_SECRET_NAME")
region = os.getenv("AWS_REGION")

# Retrieve AWS credentials from Secrets Manager
aws_access_key_id, aws_secret_access_key = get_aws_credentials_from_secrets_manager(secret_name, region)

s3 = boto3.client('s3', 
                  aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key,
                  region_name=aws_region)


# Initialize FastAPI app
app = FastAPI(
    title="Medical AI Assistant API",
    description="Backend API for generating medical reports and health recommendations.",
    version="1.0.0"
)

# Configure CORS (Cross-Origin Resource Sharing)
# to make requests to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development, allows all origins. Restrict this in production!
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], # Allow common HTTP methods
    allow_headers=["*"], # Allow all headers
)

# Pydantic model for validating the request body for the medical report endpoint
class MedicalReportRequest(BaseModel):
    sid_no: str
    visit_date: str
    branch_id: str

# --- API Endpoint 1: Generate Medical Report (/get-medical-condition) ---
@app.post("/get-medical-condition", summary="Generate Medical Report from Lab Data")
async def get_medical_condition_endpoint(request_body: MedicalReportRequest):
    """
    Fetches raw lab data from an external API, integrates it with a biomarker master table,
    saves the processed data to an Excel file, and returns the cleaned data as JSON.
    """
    try:
        sid_no = request_body.sid_no
        visit_date = request_body.visit_date
        branch_id = request_body.branch_id

        # Input validation
        if not all([sid_no, visit_date, branch_id]):
            logger.warning("Missing input fields in medical report request.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing 'sid_no', 'visit_date', or 'branch_id' in request body."
            )

        # Step 1: Fetch lab data using your medical_utils function
        api_url = f"{API_BASE_URL}?sid_no={sid_no}&visit_date={visit_date}&branch_id={branch_id}"
        lab_df = fetch_lab_data(api_url)

        if lab_df.empty:
            logger.info(f"No lab data found from external API for SID {sid_no}.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No lab data found for the provided patient details."
            )

        # Step 2: Load biomarker master data and integrate it with lab data
        biomarker_df = load_biomarker_master_table(s3,BUCKET,tactics_path,bio_marker)
        integrated_df = integrate_data(lab_df, biomarker_df)

        if integrated_df.empty:
            logger.info(f"No flagged results found after integration for SID {sid_no}.")
            # If no flagged conditions, return a success message with empty data
            return {
                "message": "Medical condition data processed. No flagged conditions found.",
                "data": []
            }

        # Step 4: Clean DataFrame (replace NaN with None for JSON serialization) and return
        clean_df = integrated_df.where(pd.notnull(integrated_df), None)

        # Upload to s3
        write_excel_to_s3(s3,BUCKET,medical_output_path,clean_df,output_key)
        
        logger.info(f"Medical report endpoint successfully processed for SID {sid_no}.")
        return {
            "message": "Medical condition data processed and saved successfully.",
            "data": clean_df.to_dict(orient="records") # FastAPI automatically serializes dicts to JSON
        }

    except DataIntegrationError as e:
        logger.error(f"Data integration specific error in /get-medical-condition: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, # Or 400 Bad Request depending on context of error
            detail=f"Data processing or integration failed: {str(e)}"
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"External API request error in /get-medical-condition: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, # Or 500 if it's an unexpected external service error
            detail=f"Failed to connect to the external lab data service: {str(e)}"
        )
    except Exception as e:
        logger.critical(f"An unhandled error occurred in /get-medical-condition: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal server error occurred: {str(e)}"
        )

# --- API Endpoint 2: Generate Recommendations (/generate-recommendation) ---
@app.post("/generate-recommendation", summary="Generate AI-based Health Recommendations")
async def generate_recommendation_endpoint():
    """
    Triggers the LangGraph agent to read the medical_output and tactic files from S3
    (via generate_recommendations) and generate comprehensive recommendations.
    """
    try:
        logger.info("Invoking LangGraph agent to generate recommendations from S3 data...")

        # The LangGraph agent internally uses generate_recommendations()
        initial_state = {}
        final_state = agent_graph.invoke(initial_state)
        recommendations = final_state.get("final_output", {})

        logger.info(f"✅ Recommendations generated successfully. Sample: {str(recommendations)[:500]}")

        # Ensure recommendations are dictionary type
        if not isinstance(recommendations, dict):
            logger.error(f"LangGraph 'final_output' is not a dictionary. Type: {type(recommendations)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"recommendations": "Invalid or missing recommendation output from LangGraph."}
            )

        return {
            "message": "Recommendations generated successfully.",
            "recommendations": recommendations}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.critical(f"Unhandled error in /generate-recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"recommendations": f"Failed to generate recommendations: {str(e)}"}
        )


# --- API Endpoint 1: Generate Medical Report (/get-medical-condition) ---
@app.post("/get-medical-condition_recommendations", summary="Generate Recommendations from Lab Data")
async def get_medical_recomendations_endpoint(request_body: MedicalReportRequest):
    """
    Fetches raw lab data from an external API, integrates it with a biomarker master table,
    saves the processed data to an Excel file, and returns the cleaned data as JSON.
    """
    try:
        sid_no = request_body.sid_no
        visit_date = request_body.visit_date
        branch_id = request_body.branch_id

        # Input validation
        if not all([sid_no, visit_date, branch_id]):
            logger.warning("Missing input fields in medical report request.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing 'sid_no', 'visit_date', or 'branch_id' in request body."
            )

        logger.info(f"Attempting to fetch lab data for SID: {sid_no}, Visit: {visit_date}, Branch: {branch_id}")

        # Fetch lab data using your medical_utils function
        api_url = f"{API_BASE_URL}?sid_no={sid_no}&visit_date={visit_date}&branch_id={branch_id}"
        lab_df = fetch_lab_data(api_url)

        if lab_df.empty:
            logger.info(f"No lab data found from external API for SID {sid_no}.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No lab data found for the provided patient details."
            )

        # Load biomarker master data and integrate it with lab data
        biomarker_df = load_biomarker_master_table(s3,BUCKET,tactics_path,bio_marker)
        integrated_df = integrate_data(lab_df, biomarker_df)

        if integrated_df.empty:
            logger.info(f"No flagged results found after integration for SID {sid_no}.")
            # If no flagged conditions, return a success message with empty data
            return {
                "message": "Medical condition data processed. No flagged conditions found.",
                "data": []}

        # Clean DataFrame (replace NaN with None for JSON serialization) and return
        clean_df = integrated_df.where(pd.notnull(integrated_df), None)

        # Upload to s3
        write_excel_to_s3(s3,BUCKET,medical_output_path,clean_df,output_key)

        # === AUTO-TRIGGER API2 (Recommendations) ===
        logger.info("Automatically triggering recommendation generation (API2).")
        initial_state = {}
        final_state = agent_graph.invoke(initial_state)
        recommendations = final_state.get("final_output", {})

        
        # === COMBINED RESPONSE ===
        return {
            # "message": "Medical condition data processed and recommendations generated.",
            # "data": clean_df.to_dict(orient="records"),
            "recommendations": recommendations
        }

    except Exception as e:
        logger.critical(f"Error in /get-medical-condition: {e}", exc_info=True)
        raise


# Optional: Root endpoint for basic API status check and documentation link
@app.get("/", include_in_schema=False)
async def read_root():
    """Provides a welcome message and a link to the API documentation."""
    return {"message": "Welcome to the Medical AI Assistant API. Visit /docs for interactive API documentation."}
