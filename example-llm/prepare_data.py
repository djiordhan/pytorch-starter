import os
import requests

def download_data():
    """
    Downloads the Tiny Shakespeare dataset for character-level language modeling.
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path = os.path.join("data", "tinyshakespeare.txt")
    
    if os.path.exists(data_path):
        print(f"Dataset already exists at {data_path}")
        return

    print("Downloading Tiny Shakespeare dataset...")
    response = requests.get(url)
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"Downloaded to {data_path}")

if __name__ == "__main__":
    download_data()
