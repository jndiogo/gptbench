
"""
Prepare a dataset with a sequence of decimal numbers starting from 0
"""

import os, sys, argparse, random
import tiktoken
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog='prepare_roman',
                        description='Prepare a dataset with a sequence of decimal=roman_literal.')

    parser.add_argument('filename', type=str, help="output filename")
    parser.add_argument('start', type=int, help="first number")
    parser.add_argument('inc', type=int, help="number increment")
    parser.add_argument('count', type=int, help="sample count")

    parser.add_argument('--gpt2', '-g', action='store_true', help="encode as gpt2 tokens")
    parser.add_argument('--sep', '-p', type=str, default=' ', help="line separator")
    parser.add_argument('--split', '-s', type=float, default=1., help="split name.ext into name.train.ext and name.val.ext at this ratio of total entries")

    
    args = parser.parse_args()

    path = args.filename
    count = args.count
    start = args.start
    inc = args.inc

    sep = args.sep
    sep=sep.replace('\\n', '\n')
    sep=sep.replace('\\t', '\t')

    split_index = int(args.split * count)

    train = []
    val = []

    index = 0 
    for n in range(start, start + count, inc):
        dest = train if index < split_index else val

        dest.append( f"{n}" )

        index += 1

    train_text = sep.join(train)
    val_text = sep.join(val)



    def save_data(path, text):

        if not len(text):
            return

        if args.gpt2:
            enc = tiktoken.get_encoding("gpt2")
            ids = enc.encode(text, allowed_special={"<|endoftext|>"})
            #ids.append(enc.eot_token) # '<|endoftext|>'

            print(f"{os.path.basename(path)}: text={len(text)} -> tokens={len(ids)} = {len(ids)/len(text)*100:.1f}%, num_vocab={enc.n_vocab}")

            with open(path, 'wb') as f:
                np.array(ids, dtype=np.uint16).tofile(f)

        else:
            print(f"{os.path.basename(path)}: text={len(text)}")

            with open(path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(text)



    if args.split != 1.:
        rest,ext = os.path.splitext(path)
        train_path = rest + ".train" + ext
        val_path = rest + ".val" + ext

        save_data(train_path, train_text)
        save_data(val_path, val_text)

    else: # only train dataset
        save_data(path, train_text)

