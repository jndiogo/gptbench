
"""
Download wikitext-2-raw-v1 dataset

"""


import os, sys, argparse, random
from pathlib import Path
import requests
import zipfile


BASE_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext"
URL=BASE_URL + "/" + "wikitext-2-raw-v1.zip"

FILES = ['wikitext-2-raw/wiki.train.raw',
         'wikitext-2-raw/wiki.test.raw', 
         'wikitext-2-raw/wiki.valid.raw']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download wikitext-2-raw dataset.')

    data_folder = os.path.join(str(Path(__file__).resolve().parent), '../data')

    parser.add_argument('-folder', type=str, default=data_folder, help="destination folder")

    args = parser.parse_args()


    # files already downloaded?
    if all([os.path.exists( os.path.join(args.folder, p) ) for p in FILES]):
        print(f"Dataset folder/files were already downloaded to {args.folder}")
        sys.exit(0)


    # download zip
    zip_filename = URL.split('/')[-1] 
    print(f'Downloading {zip_filename}')
    req = requests.get(URL) 
    with open(zip_filename,'wb') as f:
        f.write(req.content)

    print(f'Extracting zip files to {args.folder}')
    with zipfile.ZipFile(zip_filename, 'r') as zip:
        zip.extractall(args.folder)

    os.remove(zip_filename)
    