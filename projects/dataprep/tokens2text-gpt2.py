"""
Token-encode text into gpt2 encoded tokens
"""

import os, sys
import tiktoken
import numpy as np

if __name__ == '__main__':

    assert len(sys.argv) >= 3

    tok_path = sys.argv[1]
    char_path = sys.argv[2]

    with open(tok_path, 'rb') as f:
        ids = np.fromfile(f, dtype=np.uint16)

    enc = tiktoken.get_encoding("gpt2")

    if ids[-1] == enc.eot_token: # '<|endoftext|>'
        ids = ids[:-1]

    text = enc.decode(ids)

    print(f"tokens={len(ids)} -> text={len(text)} = {len(ids)/len(text)*100:.1f}%", flush=True)


    with open(char_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
