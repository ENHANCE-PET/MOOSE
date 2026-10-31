import os
import tempfile
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
    temporary_path = None
    try:
        with requests.get(item_url, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("Content-Length", 0))
            with output_manager.create_file_progress_bar() as progress:
                task = progress.add_task(f"[white] Downloading {item}...", total=total_size)
                # A private file in the destination directory avoids following a
                # pre-existing destination symlink and allows an atomic replace.
                with tempfile.NamedTemporaryFile(dir=os.path.dirname(os.path.abspath(download_file_path)),
                                                 prefix=".enhance-", delete=False) as target:
                    temporary_path = target.name
                    for chunk in response.iter_content(chunk_size=1024 * 10):
                        if chunk:
                            target.write(chunk)
                            progress.update(task, advance=len(chunk))
        os.replace(temporary_path, download_file_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            os.unlink(temporary_path)


def get_default_download_folder():
    if os.name == 'nt':  # For Windows
        download_folder = Path(os.getenv('USERPROFILE')) / 'Downloads'
    else:  # For macOS and Linux
        download_folder = Path.home() / 'Downloads'

    return download_folder
