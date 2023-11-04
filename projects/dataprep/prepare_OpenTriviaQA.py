"""
Convert an OpenTriviaQA dataset (CC-BY-SA-4.0 license) into a text file we can use to train GPT.
Download the source dataset from:
 https://github.com/uberspot/OpenTriviaQA/

This dataset is available as several plain text files, one per category. An extry example:

#Q What is the largest and most populous continent on the planet?
^ Asia
A North America
B Africa
C Asia
D South America


From each entry, such as the above, we'll generate Q: ... A: ... <|endoftext|>:
Q: What is the largest and most populous continent on the planet?
A: Asia<|endoftext|>


Contents massaging:
Some files have bad utf-8 chars. The easiest solution is to open in a text editor like Sublime Text and do a Save with Encoding > utf-8

Will be skipping any questions that include the text "one of the following" as these are multiple-choice entries. We could list all the choices but this complicates learning as it deviates from Q: A: structure.

Skip questions with " of these " which depend on listed choices.

Skip questions with "this": #Q Snakes consume their food by means of this process.

Skip: #Q These are typical Australian animals except one.

Skip: #Q Find the untrue statement about Gregory Peck.

Most questions that end with '.'' can end with '?'. It's tricky:
#Q Michael Jackson was born in this US state on August 29, 1958.



“hippopotamus” -> "hippopotamus"




#Q We know very little about the life of the mathematician Diophantus (called the father of algebra) except that he lived around the year 250 B.C. Due to one admirer of his, who described his life by the means of an algebraic riddle, we can at least determine his age at death:



"""

import os, sys, argparse, json, random, re
from pathlib import Path
import numpy as np
import tiktoken

EOT_MARKER = "<|endoftext|>"



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog='prepare_arc',
                        description='Prepare a dataset with ARC data.')

    parser.add_argument('folder_path', type=str, help="path of the source categories/ folder")
    parser.add_argument('out_path', type=str, help="output filename")

    parser.add_argument('--shuffle', '-u', action='store_true', help="order shuffle")

    parser.add_argument('--split', '-s', type=float, default=1., help="split name.ext into name.train.ext and name.val.ext at this ratio of total entries")

    parser.add_argument('--gpt2', '-g', action='store_true', help="encode as gpt2 tokens")

    
    args = parser.parse_args()


    # read all files in folder

    def try_read(file):
        encodings = ['utf-8'] # , 'cp1252', 'windows-1250', 'windows-1252'
        for e in encodings:
            try:
                read = child.read_text(encoding=e)
                #print(f"  -> Read as {e}")
                return read
            except UnicodeDecodeError as u:
                print(f"Encoding {e} -> {u}")

        assert False


    in_text = ''
    for child in Path(args.folder_path).iterdir():
        if child.is_file():
            print(f"Slurping {child.name}")

            read = try_read(child)

            in_text += read

    lines = in_text.split('\n')

    # massaging
    out = []

    count = 0
    skipping = False
    for l in lines:
        if skipping:
            if l[:2] != '#Q':
                continue
            else:
                skipping = False

        if l == '':
            continue

        if l[:2] == '#Q':
            skips = ['the following', 'of these', 'this', 'these', 'which statement', '#Q Find']
            lc = l.lower()
            if any([s in lc for s in skips]):
                skipping = True
                continue

            count+=1
            out.append('') # empty line before this Q

        # “hippopotamus” -> "hippopotamus"
        l = l.replace('“', '"')
        l = l.replace('”', '"')
        l = l.replace('  ', ' ')

        out.append(l.strip())

    print(f"QAs mid-count: {count}")

    mid = '\n'.join(out)

    """
    with open('mid.txt', 'w', encoding='utf-8') as file:
        file.write(mid)
    """

    p = re.compile(r'^#Q ([^^]+)^\^ (.+)', re.MULTILINE)

    mats = p.findall(mid)

    print(f"Matching entries count: {len(mats)}")


    split_index = int(args.split * len(mats))

    if args.shuffle:
        random.shuffle(mats)

    train=[]
    val=[]

    for i,m in enumerate(mats):
        #print(m[0])
        q = m[0].strip()
        a = m[1].strip()

        q = q.replace('\n', ' ')

        text = 'Q: ' + q + '\n'
        text += 'A: ' + a + EOT_MARKER

        if i < split_index:
            train.append(text)
        else:
            val.append(text)



    def save_data(path, text):

        if not len(text):
            return

        if args.gpt2:
            enc = tiktoken.get_encoding("gpt2")
            ids = enc.encode(text, allowed_special={EOT_MARKER})
            #ids.append(enc.eot_token) # '<|endoftext|>'

            print(f"{os.path.basename(path)}: text={len(text)} -> tokens={len(ids)} = {len(ids)/len(text)*100:.1f}%, num_vocab={enc.n_vocab}")

            with open(path, 'wb') as f:
                np.array(ids, dtype=np.uint16).tofile(f)

        else:
            print(f"{os.path.basename(path)}: len={len(text)}")

            with open(path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(text)


    train_text = ''.join(train)
    val_text = ''.join(val)

    if args.split != 1.:
        print(f"Entries - split_index: {split_index} of {len(mats)}")

        rest,ext = os.path.splitext(args.out_path)
        train_path = rest + ".train" + ext
        val_path = rest + ".val" + ext

        save_data(train_path, train_text)
        save_data(val_path, val_text)

    else: # only train dataset
        print(f"Entries: {len(mats)}")
        save_data(args.out_path, train_text)

