from huggingface_hub import upload_folder

repo_id = "sc-likes-to-code/reviewiq-bert-model"

upload_folder(
    folder_path="./saved_model",
    repo_id=repo_id,
    repo_type="model"
)

print("Model uploaded successfully.")