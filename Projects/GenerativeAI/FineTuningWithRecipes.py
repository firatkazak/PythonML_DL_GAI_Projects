import pandas as pd
from pprint import pprint
import json
from dotenv import load_dotenv
from openai import OpenAI
import time

load_dotenv(override=True)


def write_json(data_list: list, filename: str) -> None:
    """Write data to JSON file with proper error handling"""
    try:
        with open(filename, "w") as out:
            for ddict in data_list:
                jout = json.dumps(ddict) + "\n"
                out.write(jout)
    except Exception as e:
        print(f"Error writing to {filename}: {e}")


def create_user_message(row):
    return f"""Title: {row['title']}\n\nIngredients: {row['ingredients']}\n\nGeneric ingredients:"""


def prepare_example_conversation(row):
    messages = []
    messages.append({"role": "system", "content": system_message})
    user_message = create_user_message(row)
    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": row["NER"]})
    return {"messages": messages}


def wait_for_fine_tuning(client, job_id):
    """Wait for fine-tuning job to complete"""
    while True:
        response = client.fine_tuning.jobs.retrieve(job_id)
        status = response.status
        print(f"Status: {status}")

        if status == "succeeded":
            return response.fine_tuned_model
        elif status in ["failed", "cancelled"]:
            raise Exception(f"Fine-tuning failed with status: {status}")

        time.sleep(30)  # Wait 30 seconds before checking again


# Load and prepare data
recipe_df = pd.read_csv("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/cookbook_recipes_nlg_10k.csv")

system_message = """You are a helpful recipe assistant.
You are to extract the generic ingredients from each of the recipes provided."""

# Prepare training and validation data
training_df = recipe_df.loc[0:16]
validation_df = recipe_df.loc[17:20]

training_data = training_df.apply(prepare_example_conversation, axis=1).tolist()
validation_data = validation_df.apply(prepare_example_conversation, axis=1).tolist()

# Write data to files with error checking
training_file_name = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/recipe_finetune_training.json"
validation_file_name = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/recipe_finetune_validation.json"

write_json(training_data, training_file_name)
write_json(validation_data, validation_file_name)

# Initialize OpenAI client
client = OpenAI()

# Upload files
try:
    training_response = client.files.create(
        file=open(training_file_name, "rb"),
        purpose="fine-tune"
    )
    training_file_id = training_response.id
    print(f"Training file ID: {training_file_id}")

    validation_response = client.files.create(
        file=open(validation_file_name, "rb"),
        purpose="fine-tune"
    )
    validation_file_id = validation_response.id
    print(f"Validation file ID: {validation_file_id}")

    # Create fine-tuning job
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model="gpt-3.5-turbo",
        suffix="recipe-ner",
        hyperparameters={"n_epochs": 2}
    )
    job_id = response.id
    print(f"Fine-tuning job ID: {job_id}")

    # Wait for fine-tuning to complete
    fine_tuned_model_id = wait_for_fine_tuning(client, job_id)
    print(f"Fine-tuned model ID: {fine_tuned_model_id}")

    # Test the model
    if fine_tuned_model_id:
        test_row = recipe_df.iloc[0]
        test_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": create_user_message(test_row)}
        ]

        response = client.chat.completions.create(
            model=fine_tuned_model_id,  # Now we're sure this exists
            messages=test_messages,
            temperature=0,
            max_tokens=200
        )

        print("\nTest completion:")
        print(response.choices[0].message.content)

except Exception as e:
    print(f"An error occurred: {e}")
