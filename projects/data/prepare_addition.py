"""
Prepare a dataset with all permutations of a+b=c, where each number is sized according to dim param
"""

import os, sys, argparse, random
import tiktoken
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog='prepare_addition',
                        description='Prepare a dataset with all permutations of "a+b=c<|endoftext|>", where each number is sized according to dim param.')

    parser.add_argument('filename', type=str, help="output filename")
    parser.add_argument('dim', type=int, help="number dimensions: 1=0..9, 2=0..99, etc")
    parser.add_argument('--times', '-t', type=int, default=1, help="how many times to generate all the entries")
    parser.add_argument('--shuffle', '-s', action='store_true', help="order shuffle")
    parser.add_argument('--gpt2', '-g', action='store_true', help="encode as gpt2 tokens")
    args = parser.parse_args()

    path = args.filename
    dim = args.dim

    sep = '<|endoftext|>'

    out = []

    for _ in range(args.times):
        for a in range(10 ** dim):
            for b in range(10 ** dim):
                c = a+b
                out.append( f"{a}+{b}={c}{sep}" )

    if args.shuffle:
        random.shuffle(out)

    text = ''.join(out)

    if args.gpt2:
        enc = tiktoken.get_encoding("gpt2")
        ids = enc.encode(text, allowed_special={"<|endoftext|>"})
        #ids.append(enc.eot_token) # '<|endoftext|>'

        print(f"text={len(text)} -> tokens={len(ids)} = {len(ids)/len(text)*100:.1f}%, num_vocab={enc.n_vocab}")

        with open(path, 'wb') as f:
            np.array(ids, dtype=np.uint16).tofile(f)

    else:
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
