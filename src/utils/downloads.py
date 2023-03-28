from pathlib import Path
import sys

import requests


DOWNLOAD_LINKS = {
    'training_data.pkl':'https://drive.google.com/uc?export=download&id=1r83rWSleRXxMGYze4bWEUtO99zpz-IjI',
    'test_data.pkl':'https://drive.google.com/uc?export=download&id=1taDXiRmqBRDZNyELHeW2dgFpdwbfkaY2',
    'training_labels.pkl':'https://drive.google.com/uc?export=download&id=1YIJbCQg4OxksMZnciz-tibn0bj9W7cN4',
}

def check_download(path: Path, accept_unknown = False) -> Path:
    """Checks that the file is downloaded, and if not, doanloads it."""

    if not path.exists():
        # Checking that the name is known
        if path.name not in DOWNLOAD_LINKS:
            if not accept_unknown:
                raise ValueError('You asked for an unknown download!')
            return path
        
        # Downloading
        from src.utils.logs import logger
        
        logger.info(f'Downloading {path.name}')
        response = requests.get(DOWNLOAD_LINKS[path.name], stream=True)
        with path.open('wb') as f:
            dl = 0
            total_length = response.headers.get('content-length')
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\rProgression: [%s%s]" % ('=' * done, ' ' * (50-done)) )    
                sys.stdout.flush()

        sys.stdout.write('\n')
    
    return path