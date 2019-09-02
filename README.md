# nlp
## Code and processes for nlp work

### Sneap text classification work
* read_sneap_urls.py pulls about half the Sneap forum text
    * *df_pages_all_backup.pkl* has data from pull.  Note that it was actually pulled using Google colab (*read_sneap_urls.ipynb*)
* *sneap_train_textcat.py* has run with 7 possible classifications. Over 60% accuracy using spacy model. Again, this was run using Google colab (*sneap_train_textcat.ipynb*)
* Potential next steps:
    * *evaluate* function needs to be moved up and edited
    * Add more spacy cleanup (lemmatization etc)
    * Next step would be to go back to the nlp tutorial and try to add entities for things like music equipment and band names.
        * Bonus if we can pull equipment and band names from elsewhere to use to generate entities.
    * More thorough vector model? NLTK maybe?
    
### General NLP
* spacy_text_classification.py has good info on entities for topic discovery/modeling
* train_textcat.py is from https://github.com/explosion/spaCy/blob/master/examples/training/train_textcat.py
