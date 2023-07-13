import json
import os

import numpy as np
from transformers import GPT2Tokenizer


class LyricsTokenizer:
    """
    Tokenize the lyrics and save them in bin files.
    Mostly adapted from https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare/prepare.py
    """

    def __init__(self, input_file_path, output_dir):
        self.input_file_path = input_file_path
        self.output_dir = output_dir

    def tokenize_lyrics(self):
        with open(self.input_file_path, "r") as f:
            data = json.load(f)

        # Tokenize the lyrics.
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        encoded_data = []
        for song_title, lyrics in data.items():
            tokens = tokenizer.encode(lyrics, add_special_tokens=True)
            encoded_data.append(tokenizer.bos_token_id)
            encoded_data.extend(tokens)
            encoded_data.append(tokenizer.eos_token_id)

        # Split into train and test
        train_data = encoded_data[: int(len(encoded_data) * 0.9)]
        val_data = encoded_data[int(len(encoded_data) * 0.9) :]

        # Write to bin files
        train_data = np.array(train_data, dtype=np.uint16)
        val_data = np.array(val_data, dtype=np.uint16)

        train_data.tofile(os.path.join(self.output_dir, "pretrain_train.bin"))
        val_data.tofile(os.path.join(self.output_dir, "pretrain_val.bin"))


if __name__ == "__main__":
    # Get lyrics from a json file
    dir_location = os.path.split(os.path.dirname(os.path.abspath("__file__")))[0]
    input_file_path = os.path.join(dir_location, "data/lyrics.json")
    output_dir = os.path.join(dir_location, "data")

    tokenizer = LyricsTokenizer(input_file_path, output_dir)
    tokenizer.tokenize_lyrics()
