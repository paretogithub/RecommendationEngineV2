import pandas as pd
import requests
import json
from urllib.parse import urlparse, parse_qs
import os
import numpy as np # For np.nan
import io
import boto3

class DataIntegrationError(Exception):
    """Custom exception for data integration process errors."""
    pass

def fetch_lab_data(full_api_url: str, timeout: int = 30) -> pd.DataFrame:
    parsed_url = urlparse(full_api_url)
    query_params = parse_qs(parsed_url.query)

    sid_no = query_params.get('sid_no', [None])[0]
    visit_date = query_params.get('visit_date', [None])[0]
    branch_id = query_params.get('branch_id', [None])[0]

    if not all([sid_no, visit_date, branch_id]):
        raise DataIntegrationError("API URL must contain 'sid_no', 'visit_date', and 'branch_id'.")

    try:
        response = requests.get(full_api_url, timeout=timeout)
        response.raise_for_status()
        raw_json_data = response.json()

        patient_details = raw_json_data.get('patient_details', {})
        lab_data_records = raw_json_data.get('test_details', [])

        if not lab_data_records:
            return pd.DataFrame()

        lab_df = pd.DataFrame(lab_data_records)

        if 'test_name' not in lab_df.columns:
            raise DataIntegrationError("Missing 'test_name' column in lab data.")

        # Standardize test names
        lab_df['test_name'] = lab_df['test_name'].astype(str).str.upper().str.rstrip('.')

        # Add all patient details to each row
        for key, value in patient_details.items():
            lab_df[key] = value
        
        # Ensure 'gender' column exists
        if 'gender' not in lab_df.columns:
            lab_df['gender'] = None

        # Convert lab values to numeric and filter out NA / non-numeric
        if 'lab_value' in lab_df.columns:
            lab_df['lab_value'] = pd.to_numeric(lab_df['lab_value'], errors='coerce')
            lab_df = lab_df[lab_df['lab_value'].notna()].copy()

        return lab_df

    except requests.exceptions.Timeout:
        raise DataIntegrationError(f"API request timed out after {timeout} seconds.")
    except requests.exceptions.RequestException as e:
        raise DataIntegrationError(f"Failed to fetch lab data: {e}")
    except json.JSONDecodeError as e:
        raise DataIntegrationError(f"Invalid JSON response from API: {e}")
    except Exception as e:
        raise DataIntegrationError(f"Unexpected error in fetching lab data: {e}")


def clean_range_string(r):
    """
    Convert range string like '5.60 - 6.99' or '0 - 150' into numeric (min, max) with 2 decimals.
    Returns (None, None) if invalid.
    """
    if pd.isna(r):
        return None, None
    try:
        r = str(r).replace('\xa0', '').strip()
        parts = r.split('-')
        if len(parts) != 2:
            return None, None
        return round(float(parts[0].strip()), 2), round(float(parts[1].strip()), 2)
    except:
        return None, None

def preprocess_biomarker_df(biomarker_df: pd.DataFrame) -> pd.DataFrame:
    """Precompute numeric ranges for all risk categories."""
    risk_columns = [
        'Significantly_Decreased', 'Mildly_Decreased', 'Normal_range',
        'Mildly_Increased', 'Significantly_Increased'
    ]
    biomarker_df = biomarker_df.copy()
    for col in risk_columns:
        biomarker_df[[f"{col}_min", f"{col}_max"]] = biomarker_df[col].apply(
            lambda x: pd.Series(clean_range_string(x))
        )
    return biomarker_df

def integrate_data(lab_df: pd.DataFrame, biomarker_df: pd.DataFrame) -> pd.DataFrame:
    if lab_df.empty or biomarker_df.empty:
        return pd.DataFrame()

    lab_df = lab_df.copy()
    biomarker_df = preprocess_biomarker_df(biomarker_df)

    # --- Normalize gender ---
    lab_df['gender'] = lab_df.get('gender', np.nan)
    lab_df['gender'] = lab_df['gender'].astype(str).str.upper().map({'M': 'male', 'F': 'female'})

    # --- Convert lab values to float with 2 decimals ---
    lab_df['lab_value'] = pd.to_numeric(lab_df['lab_value'], errors='coerce').round(2)

    # --- Convert patient age to numeric ---
    lab_df['age'] = pd.to_numeric(lab_df.get('age', None), errors='coerce')

    # --- Initialize output columns ---
    lab_df['Risk_Category'] = None
    lab_df['Description'] = None
    lab_df['Action'] = None

    risk_columns = [
        'Significantly_Decreased', 'Mildly_Decreased', 'Normal_range',
        'Mildly_Increased', 'Significantly_Increased'
    ]

    # --- Classification function ---
    def classify_row(row):
        test_name = row.get('test_name')
        patient_age = row.get('age')
        patient_gender = row.get('gender')
        lab_value = row.get('lab_value')

        if pd.isna(lab_value):
            row['Description'] = "Invalid or missing lab value"
            row['Action'] = "Review"
            return row

        # Filter biomarker table for the test
        refs = biomarker_df[biomarker_df['Standardised_test_name'] == test_name].copy()
        if refs.empty:
            row['Description'] = f"No reference ranges for test: {test_name}"
            row['Action'] = "Review"
            return row

        # Gender filter
        if not pd.isna(patient_gender) and 'Sex' in refs.columns:
            refs = refs[
                (refs['Sex'].str.lower() == str(patient_gender).lower()) |
                (refs['Sex'].str.lower() == 'both') |
                (refs['Sex'].isna())
            ]

        # Age filter
        if not pd.isna(patient_age) and 'Age Range' in refs.columns:
            age_matches = []
            for _, r in refs.iterrows():
                age_min, age_max = clean_range_string(r.get('Age Range'))
                if age_min is None or age_max is None:
                    continue  # skip invalid ranges
                if age_min <= patient_age <= age_max:
                    age_matches.append(r)
            if age_matches:
                refs = pd.DataFrame(age_matches)

        if refs.empty:
            row['Description'] = f"No matching reference for {test_name}, age {patient_age}, gender {patient_gender}"
            row['Action'] = "Review"
            return row

        # Pick the most specific age range if multiple
        if len(refs) > 1 and 'Age Range' in refs.columns:
            refs['age_range_size'] = refs['Age Range'].apply(
                lambda x: float('inf') if pd.isna(x) else clean_range_string(x)[1] - clean_range_string(x)[0]
            )
            refs = refs.sort_values('age_range_size')

        ref = refs.iloc[0]

        # Check risk category
        for col in risk_columns:
            min_val = ref.get(f"{col}_min")
            max_val = ref.get(f"{col}_max")
            if min_val is not None and max_val is not None and min_val <= lab_value <= max_val:
                #category = col.replace("_", "_").title()
                row['Risk_Category'] = col
                row['Description'] = f"Lab value {lab_value:.2f} falls within {col} range ({min_val}-{max_val})"
                if "Significantly" in col:
                    row['Action'] = "Alert"
                elif "Mildly" in col:
                    row['Action'] = "Monitor"
                elif "Normal" in col:
                    row['Action'] = "None"
                return row

        # If lab_value doesn't fit any range
        row['Description'] = f"Lab value {lab_value:.2f} does not match any defined range"
        row['Action'] = "Review"
        return row

    # --- Apply classification to all rows ---
    final_df = lab_df.apply(classify_row, axis=1)

    final_cols = [
        'patientid', 'visit_id', 'sid_no', 'branch_id', 'visit_date', 
        'age', 'gender',  'ageinterval',
        'dept_code', 'dept_main_name', 'test_code', 'profile_code', 'test_name',
        'profile_name', 'lab_value', 
        'Risk_Category', 'Description', 'Action'
    ]

    # 'mobile_number', 'email','patient_name',

    final_df = final_df[final_cols]

    final_df = final_df[final_df['Risk_Category'].notna()].drop_duplicates()

    return final_df

## From s3
def read_excel_from_s3(s3_client,BUCKET,S3_PREFIX,key: str) -> pd.DataFrame:
    """Read Excel file from S3 and return as DataFrame"""
    obj = s3_client.get_object(Bucket=BUCKET, Key=f"{S3_PREFIX}/{key}")
    return pd.read_excel(io.BytesIO(obj["Body"].read()))

# To s3
def write_excel_to_s3(s3_client,BUCKET,S3_PREFIX, df: pd.DataFrame, key: str):
    """Write DataFrame to S3 as Excel file"""
    with io.BytesIO() as output:
        df.to_excel(output, index=False)
        output.seek(0)
        s3_client.put_object(Bucket=BUCKET, Key=f"{S3_PREFIX}/{key}", Body=output.getvalue())


# Get Aws credentials
def get_aws_credentials_from_secrets_manager(secret_name, region):
    # Create a Secrets Manager client
    client = boto3.client('secretsmanager', region_name=region)
    
    # Retrieve the secret value from Secrets Manager
    response = client.get_secret_value(SecretId=secret_name)
    
    # Extract the secret string
    secret = response['SecretString']
    
    # Parse the secret JSON string to extract AWS credentials
    credentials = json.loads(secret)
    return credentials['aws_access_key_id'], credentials['aws_secret_access_key']


def load_biomarker_master_table(s3_client, BUCKET: str, S3_PREFIX: str, key: str) -> pd.DataFrame:
    """
    Loads the biomarker master table directly from S3 (Excel).
    """

    try:
        df = read_excel_from_s3(s3_client, BUCKET, S3_PREFIX, key)

        if df.iloc[0, 4:].astype(str).str.contains('Alert|Monitor|None', na=False).any():
            df = df.iloc[1:].reset_index(drop=True)

        if 'Gender' in df.columns:
            df = df.rename(columns={'Gender': 'Sex'})

        required_cols = [
            'Standardised_test_name', 'Sex', 'Age Range',
            'Significantly_Decreased', 'Mildly_Decreased', 'Normal_range',
            'Mildly_Increased', 'Significantly_Increased'
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataIntegrationError(f"Biomarker table missing columns: {missing_cols}")

        df['Standardised_test_name'] = df['Standardised_test_name'].astype(str).str.upper()
        df['Sex'] = df['Sex'].astype(str).str.lower()

        return df

    except Exception as e:
        raise DataIntegrationError(f"Error loading biomarker table from S3: {e}")