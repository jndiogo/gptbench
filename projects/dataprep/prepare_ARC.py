"""
Convert an ARC dataset into a text file we can use to train GPT.
Download the source dataset from:
 https://allenai.org/data/arc

This dataset is available in CSV and JSON versions. An example entry of the JSON version:

{"id":"Mercury_7071540","question":{"stem":"An organism, such as a nematode worm, may have only 1000 cells. It should be classified as being","choices":[{"text":"a virus.","label":"A"},{"text":"a bacteria.","label":"B"},{"text":"unicellular.","label":"C"},{"text":"multicellular.","label":"D"}]},"answerKey":"D"}

From each entry, such as the above, we'll generate 3 variations delimited by Q: ... A: ... <|endoftext|>:

Q: An organism, such as a nematode worm, may have only 1000 cells. It should be classified as being
1: a virus.
2: a bacteria
3: unicellular
4: multicellular
A: 4<|endoftext|>
--------------------
Q: An organism, such as a nematode worm, may have only 1000 cells. It should be classified as being
- a virus.
- a bacteria
- unicellular
- multicellular
A: multicellular<|endoftext|>
--------------------
Q: An organism, such as a nematode worm, may have only 1000 cells. It should be classified as being
A: multicellular<|endoftext|>
--------------------

The '--------------------' separator is not included in text file.
"""


import os, sys, argparse, json, random
import numpy as np
import tiktoken

EOT_MARKER = "<|endoftext|>"



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog='prepare_arc',
                        description='Prepare a dataset with ARC data.')

    parser.add_argument('arc_jsonl_path', type=str, help="path of the source ARC .jsonl file folder, one of '.../ARC-Easy/'' or '.../ARC-Challenge/'")
    parser.add_argument('out_path', type=str, help="output filename")

    parser.add_argument('--variants', '-v', type=int, default=1|2|4, help="mask for the three variants: 1=Q: 1,2, A: 1 2=Q: -t1 -t2 A: t1, 4: Q: text1 A: text2")

    parser.add_argument('--gpt2', '-g', action='store_true', help="encode as gpt2 tokens")

    parser.add_argument('--shuffle', '-u', action='store_true', help="order shuffle")
    
    args = parser.parse_args()

    jsonl_path = args.arc_jsonl_path
    out_path = args.out_path


    with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file:
        json_list = list(jsonl_file)


    out = []

    for json_str in json_list:
        entry = json.loads(json_str)

        q = entry['question']

        question = q['stem']

        choices = [(c['text'], c['label']) for c in q['choices']]

        answer_index = None
        for i,c in enumerate(choices):
            if c[1] == entry['answerKey']:
                answer_index = i
                break

        assert answer_index is not None, "Bad answer key at id=" + entry['id']

        """
        print(entry)
        print(question)
        print(choices)
        print(answer_index)
        """

        """
        if not question.endswith('?'):
            question += '...'
        """

        answer_text = choices[answer_index][0]

        # Q: 1,2, A: 1
        if args.variants & 1:

            qa = f"Q: {question}\n"

            for i,c in enumerate(choices):
                qa += f"{i+1}: {choices[i][0]}\n"

            qa += f"A: {answer_index+1}{EOT_MARKER}"
        #print(qa)

        out.append(qa)


        # Q: -t1 -t2 A: t1
        if args.variants & 2:

            qa = f"Q: {question}\n"

            for i,c in enumerate(choices):
                qa += f"- {choices[i][0]}\n"

            qa += f"A: {answer_text}{EOT_MARKER}"
        #print(qa)

        out.append(qa)


        # Q: text1 A: text2
        if args.variants & 2:

            qa = f"Q: {question}\n"

            qa += f"A: {answer_text}{EOT_MARKER}"
        #print(qa)

        out.append(qa)



        # sys.exit(0)



    if args.shuffle:
        random.shuffle(out)



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



    out_text = ''.join(out)

    print(f"Entries: {len(out)}")

    save_data(out_path, out_text)

