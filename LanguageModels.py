import json
import os
import openai
import anthropic
import sys

# Remove unused imports
# from IPython.display import clear_output

# Load secrets (still reading from a local file — adapt if desired)
with open("C:/Users/HP/Python/Jupyter Notebook/CustomLibs/LLM/secret.json") as file:
    keys = json.load(file)


def Help() -> None:
    text = """
    LLMs Custom Library
    ------------------
    Available functions:

        CallLLM(assistant_prompt: str, user_prompt: str, model='gpt-4o', temp=0.0) -> dict
        
        BatchUploader(df, id_col, inf_col, role, question, label_col=None, model='gpt-4o', max_tokens=1000, description='', path=None) -> Batch
        
        BatchChecker(batch_id) -> Batch
        
        BatchRetriever(df, id_col, file_path) -> DataFrame
    """
    print(text)


def CallLLM(assistant_prompt, user_prompt, labels=None, model="gpt-4o", max_tokens=2000, temp=0.0) -> dict:
    if model == "gpt-4o":
        input_cost = 2.50 / 1_000_000
        output_cost = 10.00 / 1_000_000
    elif model == "gpt-4o-mini":
        input_cost = 0.15 / 1_000_000
        output_cost = 0.60 / 1_000_000
    elif model == "claude-3-5-sonnet":
        model += "-20241022"
        input_cost = 3.50 / 1_000_000
        output_cost = 15.0 / 1_000_000
    else:
        print(f"ERROR: Model '{model}' doesn't exist.")
        return {}

    # Replace <\labels\> in assistant_prompt if labels are provided
    if labels is not None:
        if isinstance(labels, list):
            assistant_prompt = assistant_prompt.replace("<\\labels\\>", "; ".join(labels))
        else:
            assistant_prompt = assistant_prompt.replace("<\\labels\\>", labels)

    # GPT-based models
    if model.startswith("g"):
        openai.api_key = keys["gpt4"]
        messages = [
            {"role": "assistant", "content": assistant_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp
        )
        response_text = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

    # Claude-based models
    elif model.startswith("c"):
        cclient = anthropic.Client(api_key=keys["claude"])
        response = cclient.completions.create(
            model=model,
            max_tokens_to_sample=max_tokens,
            temperature=temp,
            prompt=anthropic.HUMAN_PROMPT + user_prompt + anthropic.AI_PROMPT
        )
        response_text = response.completion
        input_tokens = response.metrics['input_characters']
        output_tokens = response.metrics['output_characters']

    return (
        user_prompt,
        {
            "content": response_text,
            "input_tokens": input_tokens,
            "input_cost": f"${round(input_cost * input_tokens, 4)}",
            "output_tokens": output_tokens,
            "output_cost": f"${round(output_cost * output_tokens, 4)}",
            "total_cost": f"${round(input_cost * input_tokens + output_cost * output_tokens, 4)}"
        }
    )


def BatchUploader(df, id_col, inf_col, role, question, label_col=None, model="gpt-4o",
                  max_tokens=1000, temp=0.0, description="", path=None):
    """
    Automatically creates .jsonl batch files for GPT or Claude, then uploads them.
    Removes confirmation prompt to just do the upload.
    """
    if path is None:
        input_file_path = './Batches/requests.jsonl'
        output_dir = './Batches/'
    else:
        input_file_path = os.path.join(path, 'requests.jsonl')
        output_dir = path

    # If using Claude with the custom suffix
    if model == "claude-3-5-sonnet":
        model += "-20241022"

    # Prepare the data
    if model.startswith("g"):
        openai.api_key = keys["gpt4"]
        default_role = role
        with open(input_file_path, 'w', encoding='utf-8') as jsonl_file:
            for _, row in df.iterrows():
                current_role = default_role
                if label_col is not None:
                    labels = row[label_col]
                    current_role = current_role.replace("<\\labels\\>", labels)

                data = {
                    "custom_id": str(row[id_col]),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": current_role},
                            {"role": "user", "content": question + f"{row[inf_col]}"}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temp
                    }
                }
                jsonl_file.write(json.dumps(data) + '\n')

    elif model.startswith("c"):
        # For Claude
        cclient = anthropic.Client(api_key=keys["claude"])
        default_role = role
        with open(input_file_path, 'w', encoding='utf-8') as jsonl_file:
            for _, row in df.iterrows():
                current_role = default_role
                if label_col is not None:
                    labels = row[label_col]
                    current_role = current_role.replace("<\\labels\\>", labels)

                data = {
                    "id": str(row[id_col]),
                    "method": "POST",
                    "url": "/v1/messages",
                    "body": {
                        "model": model,
                        "system": current_role,
                        "messages": [
                            {"role": "user", "content": question + f"{row[inf_col]}"}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temp
                    }
                }
                jsonl_file.write(json.dumps(data) + '\n')

    # Split into multiple batches if file is over 100 MB
    max_file_size = 99 * 1024 * 1024  # 99 MB
    file_count = 1
    current_size = 0
    output_file_path = os.path.join(output_dir, f'Batch{file_count}.jsonl')
    output_file = open(output_file_path, 'w', encoding='utf-8')

    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            line_size = len(line.encode('utf-8'))
            if current_size + line_size > max_file_size:
                output_file.close()
                file_count += 1
                output_file_path = os.path.join(output_dir, f'Batch{file_count}.jsonl')
                output_file = open(output_file_path, 'w', encoding='utf-8')
                current_size = 0
            output_file.write(line)
            current_size += line_size

    output_file.close()

    print("Batches ready for upload. Uploading...")

    # Auto-upload the first batch
    if model.startswith("g"):
        # GPT models
        client = openai
        batch_input_file = client.File.create(
            file=open(os.path.join(output_dir, "Batch1.jsonl"), "rb"),
            purpose="batch"
        )
        print("Batch sent.")
        batch = client.Batch.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description}
        )
        print(f"Process completed! Batch ID is: {batch.id}")
        print("Call BatchChecker(Batch ID) to check status.")
        return batch

    elif model.startswith("c"):
        # Claude models
        cclient = anthropic.Client(api_key=keys["claude"])
        batch_input_file = cclient.files.create(
            file=open(os.path.join(output_dir, "Batch1.jsonl"), "rb")
        )
        print("Batch sent.")
        batch = cclient.batches.create(
            input_file=batch_input_file.id,
            max_attempts=1,
            metadata={"description": description}
        )
        print(f"Process completed! Batch ID is: {batch.id}")
        print("Call BatchChecker(Batch ID) to check status.")
        return batch

    return {}


def BatchChecker(batch_id):
    """ Checks the status of a GPT batch and downloads the output if complete. """
    openai.api_key = keys["gpt4"]
    retrieved = openai.Batch.retrieve(batch_id)
    print(f"Batch status retrieved.\nStatus: {retrieved.status}.\nRequest counts: {retrieved.request_counts}")
    if retrieved.status == "completed":
        print("Process has been completed! Initializing result retrieving...")
        file_response = openai.File.download(retrieved.output_file_id)
        content_text = file_response.decode('utf-8')
        print("Saving in txt format.")
        with open('C:/Users/HP/downloads/output.txt', 'w', encoding='utf-8') as outfile:
            outfile.write(content_text)
        print("File created in Downloads.")
    return retrieved


def BatchCheckerExp(batch_id, model):
    """ Checks the status of a batch (GPT or Claude) and downloads output if complete. """
    if model.startswith("g"):
        openai.api_key = keys["gpt4"]
        retrieved = openai.Batch.retrieve(batch_id)
        print(f"Batch status retrieved.\nStatus: {retrieved.status}.\nRequest counts: {retrieved.request_counts}")

        if retrieved.status == "completed":
            print("Process has been completed! Initializing result retrieving...")
            file_response = openai.File.download(retrieved.output_file_id)
            content_text = file_response.decode('utf-8')
            print("Saving in txt format.")
            with open('C:/Users/HP/downloads/output.txt', 'w', encoding='utf-8') as outfile:
                outfile.write(content_text)
            print("File created in Downloads.")

    elif model.startswith("c"):
        cclient = anthropic.Client(api_key=keys["claude"])
        retrieved = cclient.batches.retrieve(batch_id)
        print(f"Batch status retrieved.\nStatus: {retrieved.status}")

        if retrieved.status == "completed":
            print("Process has been completed! Initializing result retrieving...")
            # Get all events from the batch
            events = cclient.batches.list_events(batch_id=batch_id)

            # Extract completed responses
            content_text = ""
            for event in events.data:
                if event.type == "message":
                    content_text += f"ID: {event.message.id}\n"
                    content_text += f"Content: {event.message.content}\n"
                    content_text += "-" * 50 + "\n"

            print("Saving in txt format.")
            with open('C:/Users/HP/downloads/output.txt', 'w', encoding='utf-8') as outfile:
                outfile.write(content_text)
            print("File created in Downloads.")

    return retrieved


def BatchRetriever(df, id_col, file_path):
    """
    Reads a local output.txt file produced by BatchChecker,
    merges responses back into DataFrame based on the given id_col.
    """
    df["response"] = None
    with open(file_path, "r", encoding='utf-8') as file:
        for row in file:
            data = json.loads(row)
            custom_id = data["custom_id"]
            content = data["response"]["body"]["choices"][0]["message"]["content"]
            df.loc[df[id_col].astype(str) == custom_id, "response"] = content

    print("Process completed.")
    return df


def ClassificaTRON3000():
    pass
