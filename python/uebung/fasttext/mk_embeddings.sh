#!/bin/sh

fasttext print-word-vectors /Users/ex47emin/Project/DSM/models/FastText/import/cc.de.300.bin < imdb_words.txt | gzip > imdb_embeddings.txt.gz
gzip -cd imdb_embeddings.txt.gz | wc
