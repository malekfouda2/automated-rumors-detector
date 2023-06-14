import torch
import socket
from transformers import BertTokenizerFast
def preprocess_input(text):
    tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased")
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, pad_to_max_length=True)
    input_ids= torch.tensor(inputs['input_ids']).unsqueeze(0)
    attention_mask=torch.tensor(inputs['attention_mask']).unsqueeze(0)
    return input_ids, attention_mask

REMOTE_SERVER = "www.google.com"

def check_internet_connection():
    try:
        # resolve the remote server's IP address
        host = socket.gethostbyname(REMOTE_SERVER)
        # attempt to connect to the remote server's HTTP port
        socket.create_connection((host, 80), 2)
        return True  # if the connection is successful, return True
    except:
        pass  # if the connection fails, ignore the exception and return False
    return False



