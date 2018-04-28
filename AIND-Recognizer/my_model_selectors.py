import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        
        best_model = None
        lowest_score = float("Inf")
        
        for state_num in range(self.min_n_components, self.max_n_components + 1):
            
            try:
                # BIC = -2 * logL + p * logN
                # L is likelihood of the fitted model
                # p is number of parameter
                # N is number of datapoint
                
                model = self.base_model(state_num)
                logL = model.score(self.X, self.lengths)
                
                datapoint_num = sum(self.lengths)
                logN = np.log(datapoint_num)
                
                # Number of parameter
                # Initial state occupation probabilities - number of states - 1
                # Transition matrix - (number of states) * (number of states - 1)
                # Emission matrix - 2 * number of states * number of features 
                # Initial state occupation probabilities + transition matrix + emission matrix
                # (number of states) ** 2 + 2 * (number of state) * (number of features) - 1
                p = state_num ** 2 + 2 * state_num * model.n_features - 1
                
                bic_score = -2 * logL + p * logN
                
                if bic_score < lowest_score:
                    lowest_score = bic_score
                    best_model = model
                
            except:
                pass
        
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        
        best_model = None
        most_discriminative_score = float("-Inf")
        
        for state_num in range(self.min_n_components, self.max_n_components + 1):
            
            try:
                model = self.base_model(state_num)

                # List for SUM(log(P(X(all but i))
                logP_X_not_i = []

                # Anti words
                anti_words = list(self.hwords.keys())
                anti_words.remove(self.this_word)

                for anti_word in anti_words:
                    anti_X, anti_lengths = self.hwords[anti_word]
                    logP_X_not_i.append(model.score(anti_X, anti_lengths))
                
                # log(P(X(i))
                logP_X_i = model.score(self.X, self.lengths)
                # 1/(M-1)SUM(log(P(X(all but i))
                anti_mean_prob = np.mean(logP_X_not_i)
                
                dic_score = logP_X_i - anti_mean_prob
                # print(dic_score)
                
                if dic_score > most_discriminative_score:
                    most_discriminative_score = dic_score
                    best_model = model
                
            except:
                pass
        
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # TODO implement model selection using CV
        
        split_method = KFold(n_splits = 2)

        best_model = None
        lowest_likelihood = float("Inf")
        
        for state_num in range(self.min_n_components, self.max_n_components + 1):
            
            try:                
                log_likelihoods = []
                
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    model = self.base_model(state_num)
                    log_likelihoods.append(model.score(test_X, test_lengths))
            
                avg_log_likelihood = np.mean(log_likelihoods)

                # print("This word is {} and number of state is {} with average log likelihood {}". \
                #     format(self.this_word, state_num, avg_log_likelihood))
            
                # The lower the average log likelikhood is the better the model is
                if avg_log_likelihood < lowest_likelihood:
                    lowest_likelihood = avg_log_likelihood
                    best_model = model
            
            except:
                pass
        
        return best_model
            
        
        
