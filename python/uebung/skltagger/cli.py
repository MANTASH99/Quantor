"""
SKLTagger: Command-line interface
"""

import pandas as pd
import argparse
from somajo import SoMaJo

from .utils import sentences2dataframe, load_model

def tag_text(pipeline, text, language='de'):
    """" Tokenize a text and POS-tag it with the specified pipeline.
    
    Parameters
    ----------
    pipeline : object
        Trained SKLTagger pipeline (should be loaded and verified with load_model())
    text : str
        Text to be tokenized and tagged specified as a single character string.
    language : str
        Two-letter ISO code indicating language of the text for tokenization.
        Languages supported by SoMaJo are German (``de``) and English (``en``).

    Returns
    -------
    tokens : DataFrame
        Pandas data frame with one token per line in column ``word``, corresponding POS tags in column ``pos``, 
        and consecutive sentence numbers in column ``sent``.
    """
    if language == 'de':
        tokenizer = SoMaJo('de_CMC', split_sentences=True)
    elif language == 'en':
        tokenizer = SoMaJo('en_PTB', split_sentences=True)
    else:
        raise ValueError("sorry, our tokenizer doesn't support language '{}'".format(language))
    sentences = [ [t.text for t in s] for s in tokenizer.tokenize_text([text]) ]
    df = sentences2dataframe(sentences)
    df['pos'] = pipeline.predict(df)
    return df

def print_vrt(df):
    """ Print output of tag_text() in vertical text format (one token per line).
    
    Parameters
    ----------
    df : DataFrame
        Pandas data frame returned by tag_text().
    """
    def print_sent(sent):
        print("<s>")
        print("\n".join(sent.word + "\t" + sent.pos))
        print("</s>")
        
    df.groupby('sent').apply(print_sent)
    
    
## command-line script (when module is loaded as main program)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='POS-tag text files or user input with a Scikit-Learn Tagger pipeline. The text is written to standard output in vertical text format with <s> tags indicating sentence breaks.')
    ap.add_argument('model', type=str, 
                      help='Disk file containing trained SKLTagger pipeline.')
    ap.add_argument('--language', '--lang', '-l', type=str, choices=('de', 'en'), default='de',
                      help='Language of input text (must match the SKLTagger pipeline).')
    ap.add_argument('--input', '-i', type=str, action='append',
                      help='Text file to be tokenized and tagged (can be used multiple times).  If unspecified, sentences can be entered interactively by the user.')
    args = ap.parse_args()

    tagger = load_model(args.model)
    
    if args.input is None:
        while True:
            sent = input("Please enter a sentence to be tagged (or RET to exit)\n> ").strip()
            if sent == "":
                break
            print_vrt(tag_text(tagger, sent, language=args.language))
    else:
        for file in args.input:
            with open(file, 'r') as fh:
                text = fh.read().strip()
                print_vrt(tag_text(tagger, text, language=args.language))
