import requests
import re
from tqdm import tqdm


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        a = re.search(r'.*confirm=([0-9A-Za-z_]+).*', response.text)
        try:
            return a[1]
        except:
            return ""

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            itr = response.iter_content(CHUNK_SIZE)
            iterat = tqdm(itr)
            for i, chunk in enumerate(iterat):
                iterat.set_postfix(
                    size=f"{round((i+1)*32768*9.53674316406*10**-7, 2)}MB")
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python google_drive.py drive_file_id destination_file_path")
    else:
        # TAKE ID FROM SHAREABLE LINK
        file_id = sys.argv[1]
        # DESTINATION FILE ON YOUR DISK
        destination = sys.argv[2]
        download_file_from_google_drive(file_id, destination)
