import py7zr
from pathlib import Path


CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / '..' / 'data'


def extract_olist():
    filepath = DATA_DIR / 'olist.7z'
    with py7zr.SevenZipFile(filepath, mode='r') as z:
        z.extractall(DATA_DIR)


if __name__ == '__main__':
    extract_olist()
