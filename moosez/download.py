import os
from pathlib import Path
import requests
from typing import Union
from moosez import constants
from moosez import system


def download_enhance_data(download_directory: Union[str, None], output_manager: system.OutputManager):
    output_manager.log_update(f"    - Downloading ENHANCE 1.6k data")
    if not download_directory:
        download_directory = get_default_download_folder()


    for item in constants.ENHANCE_URLS.keys():
        download_file_name = os.path.basename(item)
        download_file_path = os.path.join(download_directory, download_file_name)
        download_enhance_item(item, download_file_path, output_manager)
    output_manager.console_update(f"{constants.ANSI_GREEN} ENHANCE 1.6k data successfuly downloaded. {constants.ANSI_RESET}")





def download_enhance_item(item, download_file_path, output_manager: system.OutputManager):

    item_url = constants.ENHANCE_URLS[item]
    response = requests.get(item_url, stream=True)
    if response.status_code != 200:
        output_manager.console_update(f"    X Failed to download {item} from {item_url}")
        raise Exception(f"Failed to download {item} from {item_url}")

    total_size = int(response.headers.get("Content-Length", 0))
    chunk_size = 1024 * 10

    progress = output_manager.create_file_progress_bar()
    with progress:
        task = progress.add_task(f"[white] Downloading {item}...", total=total_size)
        with open(download_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress.update(task, advance=chunk_size)


def get_default_download_folder():
    if os.name == 'nt':  # For Windows
        download_folder = Path(os.getenv('USERPROFILE')) / 'Downloads'
    else:  # For macOS and Linux
        download_folder = Path.home() / 'Downloads'

    return download_folder
