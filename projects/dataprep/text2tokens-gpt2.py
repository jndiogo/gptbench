"""
Token-encode text into gpt2 encoded tokens
"""

import os, sys
import tiktoken
import numpy as np

if __name__ == '__main__':

    assert len(sys.argv) >= 3

    char_path = sys.argv[1]
    tok_path = sys.argv[2]

    with open(char_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    #text=text[:20]

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(text)
    ids.append(enc.eot_token) # '<|endoftext|>'

    print(f"text={len(text)} -> tokens={len(ids)} = {len(ids)/len(text)*100:.1f}%, num_vocab={enc.n_vocab}", flush=True)

    with open(tok_path, 'wb') as f:
        np.array(ids, dtype=np.uint16).tofile(f)
