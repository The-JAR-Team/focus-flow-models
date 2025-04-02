import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv('API_KEY')
FOLDER_ID = os.getenv('FOLDER_ID')


def list_files(folder_id, api_key):
    """
    Lists all files in the specified Google Drive folder using the API key.
    """
    url = "https://www.googleapis.com/drive/v3/files"
    params = {
        'q': f"'{folder_id}' in parents and trashed=false",
        'key': api_key,
        'fields': 'files(id, name)'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('files', [])
    else:
        print("Error listing files:", response.text)
        return []


def download_file(file_id, destination):
    """
    Downloads a single file from Google Drive using its file ID.
    """
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {destination}")
    else:
        print("Error downloading file:", response.text)


def download_folder(folder_id, api_key, path):
    """
    Downloads all files from the specified folder into the given directory.
    Skips files that already exist.
    """
    files = list_files(folder_id, api_key)

    if not os.path.exists(path):
        os.makedirs(path)

    for file in files:
        file_id = file['id']
        file_name = file['name']
        destination = os.path.join(path, file_name)

        if os.path.exists(destination):
            print(f"Skipping {file_name}: already exists.")
        else:
            download_file(file_id, destination)


def get_download_location():
    """
    Prompts the user for the download location.
    If no input is provided, defaults to './Data'.
    """
    user_input = input("Enter the directory path to download files to (default: ./Data): ").strip()
    return user_input if user_input else "./Data"


if __name__ == '__main__':
    download_path = './dataset'
    download_folder(FOLDER_ID, API_KEY, download_path)
