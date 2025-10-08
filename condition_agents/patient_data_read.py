import pandas as pd
from typing import Dict, Any
import boto3

from medical_utils import read_excel_from_s3, get_aws_credentials_from_secrets_manager
from config import *

# Retrieve AWS credentials from Secrets Manager
aws_access_key_id, aws_secret_access_key = get_aws_credentials_from_secrets_manager(secret_name, aws_region)

s3 = boto3.client('s3', 
                  aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key,
                  region_name=aws_region)


# def patient_data_node(state: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Reads medical data from S3 and updates patient_data in state.
#     """
#     try:
#         # Read Excel directly from S3
#         df = read_excel_from_s3(s3, BUCKET, medical_output_path, output_key)

#         cols = ["patientid", "visit_id", "age", "gender", "ageinterval", "test_name", "Risk_Category"]
#         df = df[df["Risk_Category"] != 'Normal_range']
#         df = df[cols].drop_duplicates()

#         # Convert to JSON/dict for downstream use
#         state["patient_data"] = df.to_dict(orient="records")
#         print(f"✅ patient_data loaded: {len(state['patient_data'])} records")

#         return state

#     except Exception as e:
#         print(f"❌ Error reading patient data from S3: {e}")
#         state["patient_data"] = None
#         return state


def patient_data_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reads medical data from S3 and creates a structured patient profile 
    containing demographics + all abnormal lab results.
    """
    try:
        # Read Excel directly from S3
        df = read_excel_from_s3(s3, BUCKET, medical_output_path, output_key)

        cols = ["patientid", "visit_id", "age", "gender", "ageinterval", "test_name", "Risk_Category"]
        df = df[df["Risk_Category"] != 'Normal_range']
        df = df[cols].drop_duplicates()

        if df.empty:
            print("⚠️ No abnormal lab records found for patient.")
            state["patient_data"] = []
            return state

        # -------------------------------
        # Step 1: Build Lab → Risk mapping
        # -------------------------------
        labs_dict = (
            df[["test_name", "Risk_Category"]]
            .dropna()
            .set_index("test_name")["Risk_Category"]
            .to_dict()
        )

        # -------------------------------
        # Step 2: Extract Demographics
        # -------------------------------
        first_row = df.iloc[0].to_dict()
        patient_profile = {
            "patientid": first_row.get("patientid"),
            "visit_id": first_row.get("visit_id"),
            "age": first_row.get("age"),
            "gender": first_row.get("gender"),
            "ageinterval": first_row.get("ageinterval"),
            "Labs": labs_dict
        }

        # -------------------------------
        # Step 3: Save in state
        # -------------------------------
        state["patient_data"] = [patient_profile]
        print(f"✅ patient_data structured successfully with {len(labs_dict)} abnormal labs")

        return state

    except Exception as e:
        print(f"❌ Error reading patient data from S3: {e}")
        state["patient_data"] = None
        return state
