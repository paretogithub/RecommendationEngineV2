from LLM_utils import generate_recommendations
import logging

# -------------------------
# Configure Logging
# -------------------------
logger = logging.getLogger("medical_app")
logger.setLevel(logging.INFO)  # INFO, DEBUG, WARNING, ERROR

# StreamHandler outputs logs to stdout (CloudWatch will capture this)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


#exercise_node
def exercise_node(state):
    logger.info("🏃 Running Exercise Node...")
    df = generate_recommendations("exercise").drop_duplicates()
    state["exercise_output"] = df.to_dict(orient="records")
    logger.info(f"exercise_output: {len(state['exercise_output'])} recommendations uploaded.")
    #print("exercise_output:",state["exercise_output"])
    return state

#nutrition_node
def nutrition_node(state):
    logger.info("🏃 Running Nutrition Node...")

    # Only take category output from generate_recommendations
    nutrition_df = generate_recommendations("nutrition")
    nutrition_df = nutrition_df.drop_duplicates()
    state["nutrition_output"] = nutrition_df.to_dict(orient="records")

    logger.info(f"nutrition_output: {len(state['nutrition_output'])} recommendations uploaded.")
    #print("nutrition_output:",state["nutrition_output"])
    return state

#supplement_node
def supplement_node(state):
    logger.info("🏃 Running Supplements Node...")

    # Only take category output from generate_recommendations
    supplement_df = generate_recommendations("supplement")
    supplement_df = supplement_df.drop_duplicates()
    state["supplement_output"] = supplement_df.to_dict(orient="records")

    logger.info(f"supplement_output: {len(state['supplement_output'])} recommendations uploaded.")
    #print("supplement_output:",state["supplement_output"])
    return state

#lifestyle_node
def lifestyle_node(state):
    logger.info("🏃 Running Lifestyle Node...")

    # Only take category output from generate_recommendations
    lifestyle_df = generate_recommendations("lifestyle")
    lifestyle_df = lifestyle_df.drop_duplicates()
    state["lifestyle_output"] = lifestyle_df.to_dict(orient="records")

    logger.info(f"lifestyle_output: {len(state['lifestyle_output'])} recommendations uploaded.")
    #print("lifestyle_output:",state["lifestyle_output"])

    return state
