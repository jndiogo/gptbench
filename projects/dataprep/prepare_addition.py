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
    parser.add_argument('--shuffle', '-u', action='store_true', help="order shuffle")
    parser.add_argument('--gpt2', '-g', action='store_true', help="encode as gpt2 tokens")
    parser.add_argument('--sep', '-p', type=str, default='<|endoftext|>', help="separator")
    parser.add_argument('--features', '-f', type=str, default='', help="'reverse': a+b=c and c=a+b, 'commutative': a+b=b+a")
    parser.add_argument('--split', '-s', type=float, default=1., help="split name.ext into name.train.ext and name.val.ext at this ratio of total entries")
    parser.add_argument('--zero', '-z', action='store_true', help="zero pad up to dim size")
    
    args = parser.parse_args()

    path = args.filename
    dim = args.dim

    sep = args.sep
    sep=sep.replace('\\n', '\n')
    sep=sep.replace('\\t', '\t')

    mult = 1
    if 'reverse' in args.features:
        mult+=1
    if 'commutative' in args.features:
        mult+=1

    split_index = int(args.split * ( (10 ** (dim*2)) * mult) )

    train = []
    val = []

    for _ in range(args.times):
        index = 0 
        for a in range(10 ** dim):
            for b in range(10 ** dim):
                dest = train if index < split_index else val

                c = a+b

                if args.zero:
                    fmt = '{:0' + str(dim) + 'd}'
                    sa = fmt.format(a)
                    sb = fmt.format(b)
                    sc = fmt.format(c)
                else:
                    sa = str(a)
                    sb = str(b)
                    sc = str(c)

                dest.append( f"{sa}+{sb}={sc}{sep}" )

                if 'reverse' in args.features:
                    dest.append( f"{sc}={sa}+{sb}{sep}" )

                if 'commutative' in args.features:
                    dest.append( f"{sa}+{sb}={sb}+{sa}{sep}" )

                index += mult


    if args.shuffle:
        random.shuffle(train)
        random.shuffle(val)


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



    train_text = ''.join(train)
    val_text = ''.join(val)


    if args.split != 1.:
        rest,ext = os.path.splitext(path)
        train_path = rest + ".train" + ext
        val_path = rest + ".val" + ext

        save_data(train_path, train_text)
        save_data(val_path, val_text)

    else: # only train dataset
        save_data(path, train_text)

