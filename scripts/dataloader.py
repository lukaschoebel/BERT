import os
import wget
import zipfile

from logging import INFO
from logger_factory import logger_factory

logger = logger_factory('dataloader', INFO)

def download_cola(dir_path: str, dest_folder: str='data') -> None:
    """ Downloads CoLA (Corpus of Linguistic Acceptability) dataset 
    
    Args:
        dir_path: string which defines path to current directory
        dest_folder: string which defines destination folder [optional]
    """

    dest_path = os.path.join(dir_path, f'../{dest_folder}')
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)

    url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

    if not os.path.exists('./cola_public_1.1.zip'):
        wget.download(url, './cola_public_1.1.zip')

        try:
            with zipfile.ZipFile('cola_public_1.1.zip') as z:
                z.extractall(dest_path)
                os.remove('./cola_public_1.1.zip')
        except:
            logger.info('zip extraction failed')


if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))
    download_cola(dir_path)