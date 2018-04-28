import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    for i in test_set.get_all_Xlengths().keys():
        i_X, i_lengths = test_set.get_all_Xlengths()[i]
        
        best_word = None
        best_prob = float("-Inf")
        probability_dict = {}
        
        for word, GaussianHMM in models.items():
            try:
                word_logL = GaussianHMM.score(i_X, i_lengths)
                probability_dict[word] = word_logL
            
                if word_logL > best_prob:
                    best_prob = word_logL
                    best_word = word
            except:
                probability_dict[word] = float("-Inf")
                continue
                
        probabilities.append(probability_dict)
        guesses.append(best_word)
        
    return probabilities, guesses
        
        