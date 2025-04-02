import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv('API_KEY')
FOLDER_ID = os.getenv('FOLDER_ID')


def list_files(folder_id, api_key):
    """
    Lists all items (files and subfolders) in the specified Google Drive folder.
    Returns items with their id, name, and mimeType.
    """
    url = "https://www.googleapis.com/drive/v3/files"
    params = {
        'q': f"'{folder_id}' in parents and trashed=false",
        'key': api_key,
        'fields': 'files(id, name, mimeType)'
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


def gather_download_tasks(folder_id, api_key, download_path):
    """
    Recursively scans the Google Drive folder and gathers tasks for files to download.
    Returns a list of tuples: (file_id, destination, file_name).
    Creates local subdirectories as needed.
    """
    tasks = []
    items = list_files(folder_id, api_key)

    # Ensure the download path exists.
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    for item in items:
        item_id = item['id']
        item_name = item['name']
        item_mime = item.get('mimeType', '')
        destination = os.path.join(download_path, item_name)

        if item_mime == 'application/vnd.google-apps.folder':
            # If the item is a folder, recursively gather tasks.
            print(f"Scanning folder: {item_name}")
            tasks.extend(gather_download_tasks(item_id, api_key, destination))
        else:
            # Only add file task if file does not exist locally.
            if not os.path.exists(destination):
                tasks.append((item_id, destination, item_name))
            else:
                print(f"Skipping {item_name}: already exists.")
    return tasks


def get_download_location():
    """
    Prompts the user for the download location.
    If no input is provided, defaults to './Data'.
    """
    user_input = input("Enter the directory path to download files to (default: ./Data): ").strip()
    return user_input if user_input else "./Data"


def download_all_files(tasks):
    """
    Downloads each file from the tasks list and prints progress updates.
    """
    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No new files to download.")
        return

    print(f"Starting download of {total_tasks} file(s).")
    downloaded_count = 0

    for idx, (file_id, destination, file_name) in enumerate(tasks, start=1):
        download_file(file_id, destination)
        downloaded_count += 1
        percentage = (downloaded_count / total_tasks) * 100
        print(f"Progress: {downloaded_count}/{total_tasks} files downloaded ({percentage:.1f}%).\n")


if __name__ == '__main__':
    # Get the local destination folder from the user.
    download_location = './dataset'

    # Gather all tasks for files to download (including subfolders).
    tasks = gather_download_tasks(FOLDER_ID, API_KEY, download_location)

    # Download all files while printing process updates.
    download_all_files(tasks)
