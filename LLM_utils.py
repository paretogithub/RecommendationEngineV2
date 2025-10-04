import pandas as pd
import re
import numpy as np
import boto3 

from medical_utils import read_excel_from_s3, get_aws_credentials_from_secrets_manager
from config import *

# Retrieve AWS credentials from Secrets Manager
aws_access_key_id, aws_secret_access_key = get_aws_credentials_from_secrets_manager(secret_name, aws_region)

s3 = boto3.client('s3', 
                  aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key,
                  region_name=aws_region)


def generate_recommendations(tactic_category):
    # Load input files
    medical_output = read_excel_from_s3(s3,BUCKET,medical_output_path,output_key)
    test_tactic = read_excel_from_s3(s3,BUCKET,tactics_path,test_tactic_name)
    tactic_master = read_excel_from_s3(s3,BUCKET,tactics_path,tactic_master_name)

    medical_output['test_name'] = medical_output['test_name'].str.lower().str.strip()
    test_tactic['test_name'] = test_tactic['test_name'].str.lower().str.strip()
    
    # Try full match: test_name + label
    medical_output['test_key'] = (medical_output['test_name'].str.lower()+ '_' + medical_output['Risk_Category'].fillna('').str.lower())
    test_tactic['test_key'] = test_tactic['test_name'].str.lower() + '_' + test_tactic['risk_category'].fillna('').str.lower()

    # First match: both test name and label
    merged = pd.merge(medical_output, test_tactic, on='test_key', how='inner').reset_index(drop=True)
    #print(merged.columns)

    # # Second match: only test_name if tactic is still missing
    # no_match = merged[merged['tactic'].isna()].copy()
    # # print("no_match",no_match.columns)
    # #no_match.to_excel("nomatch.xlsx")
 
    # if not no_match.empty:
    #     # Ensure we have a clean test_name column to merge on
    #     no_match['test_name_clean'] = no_match['test_name_x'].str.strip().str.lower()

    #     test_tactic['test_name_clean'] = test_tactic['test_name'].str.strip().str.lower()

    #     fallback = pd.merge(
    #         no_match.drop(columns=['test_key', 'test_name_y', 'tactic', 'tactic_category'], errors='ignore'),
    #         test_tactic.drop(columns=['test_key'], errors='ignore'),
    #         on='test_name_clean',
    #         how='left'
    #     )

    #     # Keep naming consistent
    #     fallback = fallback.rename(columns={'test_name_clean': 'test_name'})


    #     # Drop duplicate column names if created by merge
    #     fallback = fallback.loc[:, ~fallback.columns.duplicated()]

    #     # Merge matched + fallback
    #     merged = pd.concat(
    #         [merged[merged['tactic'].notna()].loc[:, ~merged.columns.duplicated()], 
    #          fallback],
    #         ignore_index=True
    #     )

    merged['tactic'] = merged['tactic'].str.strip().str.lower()
    merged['tactic_category'] = merged['tactic_category'].str.strip().str.lower()
    tactic_master['tactic'] = tactic_master['tactic'].str.strip().str.lower()
    tactic_master['tactic_category'] = tactic_master['tactic_category'].str.strip().str.lower()

    # Combining patient tactic with tactic master
    merged_tactic = pd.merge(
        merged,
        tactic_master,
        left_on=['tactic_category', 'tactic'],
        right_on=['tactic_category', 'tactic'],
        how='inner'
    )

    # Filter by tactic_category
    filtered = merged_tactic[merged_tactic['tactic_category'] == tactic_category]

    final = filtered.drop_duplicates()
    final = final.where(pd.notna(final), None)

    return final[[ 'visit_id', 'tactic_category', 'recommendations','When_to_recommend','When_not_to_recommend']]



