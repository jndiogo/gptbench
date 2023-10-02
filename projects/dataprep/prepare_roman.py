def int_from_roman(s):
  """
  Based from https://www.tutorialspoint.com/roman-to-integer-in-python
  """
  roman = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000,'IV':4,'IX':9,'XL':40,'XC':90,'CD':400,'CM':900}
  i = 0
  num = 0
  while i < len(s):
     if i+1<len(s) and s[i:i+2] in roman:
        num+=roman[s[i:i+2]]
        i+=2
     else:
        #print(i)
        num+=roman[s[i]]
        i+=1
  return num



def roman_from_int(number):
    """
    Based from https://www.geeksforgeeks.org/python-program-to-convert-integer-to-roman/
    """



    num = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]
    sym = ["I", "IV", "V", "IX", "X", "XL", "L", "XC", "C", "CD", "D", "CM", "M"]


    out = ''

    i = len(num)-1     
    while number:
        div = number // num[i]
        number %= num[i]
 
        while div:
            out = out + sym[i]
            div -= 1

        i -= 1

    return out



"""
Prepare a dataset with a sequence of decimal=roman_literal
"""

import os, sys, argparse, random
import tiktoken
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog='prepare_roman',
                        description='Prepare a dataset with a sequence of decimal=roman_literal.')

    parser.add_argument('filename', type=str, help="output filename")
    parser.add_argument('count', type=int, help="sample count starting in 1")
    parser.add_argument('--shuffle', '-u', action='store_true', help="order shuffle")
    parser.add_argument('--gpt2', '-g', action='store_true', help="encode as gpt2 tokens")
    parser.add_argument('--sep', '-p', type=str, default='\\n', help="line separator")
    parser.add_argument('--split', '-s', type=float, default=1., help="split name.ext into name.train.ext and name.val.ext at this ratio of total entries")
    parser.add_argument('--rev', '-r', action='store_true', help="create roman=decimal")

    
    args = parser.parse_args()

    path = args.filename
    count = args.count

    sep = args.sep
    sep=sep.replace('\\n', '\n')
    sep=sep.replace('\\t', '\t')

    split_index = int(args.split * count)

    train = []
    val = []

    index = 0 
    for n in range(1,count+1):
        dest = train if index < split_index else val

        rom = roman_from_int(n)

        if args.rev:
            dest.append( f"{rom}={n}" )
        else:
            dest.append( f"{n}={rom}" )

        index += 1


    if args.shuffle:
        random.shuffle(train)
        random.shuffle(val)


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

