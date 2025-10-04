
# Define the AWS region and Secrets Manager secret name

BUCKET ='paretobucket'

medical_output_path = 'Production/recommendation_model/medical_output_trns'
output_key ='medical_output.xlsx'

tactics_path = 'Production/recommendation_model/master_data'
test_tactic_name = 'New_test_tactic.xlsx'
tactic_master_name ='New_tactic_master.xlsx'
bio_marker = 'Biomarker_master.xlsx'

API_BASE_URL = "https://yxy4mpab5y.ap-south-1.awsapprunner.com/fetchrecords"
