import time
from multiprocessing.pool import ThreadPool
import pickle
import logging
from datetime import datetime

# Logging setup
LOGGING = True
if LOGGING:
    log_filename = datetime.now().strftime("%m-%d_%H-%M") + "_logfile.log"
    logging.basicConfig(filename=log_filename,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

with open("evidence_preprocessed_bert_v3.pkl","rb") as f:

    evidence = pickle.load(f)
    if LOGGING:
        logging.info("evidence loaded")


# Barely speeds up running time if at all???
# Thank GIL :c definitely not I/O bound
N_THREADS = 2


# Surprisingly couldn't find an implementation on torch/tf/keras
# Byte Pair Encoding tokenizer to feed into BERT-like model below
# Some sections referenced from 
# https://martinlwx.github.io/en/the-bpe-tokenizer/
class BPE:
    def __init__(self, corpus, vocab_size, min_count=1):
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.min_count = min_count
        
        self.vocab = []
        
        self.word_freq = {}
        
        # For multiprocessing only
        self.word_freq_partitions = [{} for _ in range(N_THREADS)]
        
        # word to fragments
        self.word_partitions = {}
        
        # fragments to bigger fragments
        self.merge_rules = {}


    def train(self):

        t_start = time.time()
        for paragraph in self.corpus:
            for sentence in paragraph:
                i = 0
                for word in sentence:
                    self.word_freq[word] = self.word_freq.get(word, 0) + 1
                    self.word_freq_partitions[i % N_THREADS][word] = self.word_freq_partitions[i % N_THREADS].get(word, 0)+1
        
        alphabet = set()
        for word in self.word_freq:

            # initialize word partitions
            char_list = list(word)
            char_list = ["##" + char_list[i] if i != 0 else char_list[0] for i in range(len(char_list))]
            self.word_partitions[word] = char_list

            # construct alphabet
            # misnomer possibly since plenty of tokens will have the ## prefix
            for letter in word:
               if letter not in alphabet:
                   alphabet.add(letter)

        self.vocab = list(alphabet)
        self.vocab.sort()

        print(f"Time {time.time()-t_start:.2f} - Alphabet initialized with size {len(alphabet)}.")
        print(f"Alphabet: {self.vocab}")

        if LOGGING:
            logging.info("Alphabet constructed.")

        
        # 05/10: Any chance of multithreading part of this somehow?
        # 05/13: nope
        iter = 0
        while len(self.vocab) < self.vocab_size:

            #t_s = time.time()
            # ~4s singlethread

            # WordPiece scaling applied here already
            pair_freq = self.get_pair_freq(wordpiece=True)

            #print(f"pair_freq: {time.time()-t_s:.2f}s")
            

            if len(pair_freq) == 0:
                print(f"Time {time.time()-t_start:.2f} - No more pairs. Exiting at vocab size {len(self.vocab)}")

            
            # Gets the pair that appears most in the training set
            
            best_pair = max(pair_freq, key=pair_freq.get)

            #t_s = time.time()
            # ~3s singlethread
            self.update_word_partitions(best_pair[0], best_pair[1])
            
            
            #print(f"update_word_partitions: {time.time()-t_s:.2f}s")

            # practically instant
            self.merge_rules[best_pair] = best_pair[0] + best_pair[1][2:]
            self.vocab.append(best_pair[0] + best_pair[1][2:])

            if LOGGING:
                logging.info(f"{str.rjust(best_pair[0], 10)} + {str.rjust(best_pair[1], 10)} => {str.rjust(best_pair[0]+best_pair[1][2:], 10)}")
            # Sanity keeping every 100 iterations - whole loop takes HOURS
            iter += 1
            if iter % 100 == 0:
                print(f"Time {time.time()-t_start:.2f} - Now on iteration {iter}.")
                if LOGGING:
                    logging.info(f"Iteration {iter} completed.")

            # Save a checkpoint every 1000 iterations
            if iter % 1000 == 0:
                self.save()
        self.save()


    # Returns a pair frequency dictionary with entries:
    # pair (tuple) : freq (int)
    def get_pair_freq(self, wordpiece = True):
        pair_freq = {}
        singleton_freq = {}

        #pair_freq_partitions = []

        # Multithreading attempt
        # Doesn't work I think.. not enough compute time on colab either
        pool = ThreadPool(N_THREADS)

        def get_pair_freq_subtask(partition):

            #pair_freq_partition = {}

            for word_freq_pair in partition.items():
                word, freq = word_freq_pair
                word_partition = self.word_partitions[word]

                for i in range(len(word_partition)-1):
                    #pair_freq_partition[(word_partition[i], word_partition[i+1])] = pair_freq_partition.get((word_partition[i], word_partition[i+1]), 0) + freq
                    pair_freq[(word_partition[i], word_partition[i+1])] = pair_freq.get((word_partition[i], word_partition[i+1]), 0) + freq
                    singleton_freq[word_partition[i]] = singleton_freq.get(word_partition[i],0) + freq
                
                singleton_freq[word_partition[len(word_partition)-1]] = singleton_freq.get(word_partition[len(word_partition)-1],0) + freq
            return

        # returns None, results are already aggregated by the time all threads join.
        pool.map(get_pair_freq_subtask, self.word_freq_partitions)

        # WordPiece scaling, divide all pair frequency scores by product of singleton scores
        if wordpiece:
            for pair, freq in pair_freq.items():
                freq = freq/singleton_freq.get(pair[0])
                freq = freq/singleton_freq.get(pair[1])



        return pair_freq


    # Merge word partitions by the new pattern
    # Modify in place for performance (saves ~0.2s/iter)
    def update_word_partitions(self, left, right):
        

        # Supposedly jank if using .items() and modifying in place so this will do
        # https://stackoverflow.com/a/6777569
        for word in self.word_partitions.keys():

            partition = self.word_partitions[word]
            length = len(partition)

            i = 0
            while i < length:
                
                # Last token, do nothing
                if i+1 >= length:
                    pass
                
                # Else check for matching pattern
                # Delete the next entry if matches
                elif partition[i] == left and partition[i+1] == right:

                    # WordPiece specific: get rid of the ## prefix on right
                    partition[i] = partition[i] + partition[i+1][2:]
                    
                    del partition[i+1]
                    length -= 1 # Otherwise runs out of bounds
    
                i += 1
    

    def tokenize(self, sentence):
            # separate sentence by space then turn words into char lists
            word_partitions = [list(t) + ["</w>"] for t in sentence.split(" ")]
    
            for left, right in self.merge_rules:
                for partition in word_partitions:

                    length = len(partition)
                    
                    i = 0
                    while i < length:
                        if i+1 >= length:
                            pass

                        elif partition[i] == left and partition[i+1] == right:
                            partition[i] = partition[i] + partition[i+1]
                            
                            del partition[i+1]
                            length -= 1 # Otherwise runs out of bounds
                    
                    i += 1

            # https://stackoverflow.com/a/952952
            return [token 
                    for partition in word_partitions 
                    for token in partition]
    
    def encode_word(self, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens

    def save(self, path=""):
        VERSION = 1.4
        # Save to a model to be reused later
        with open(f"{path}BPETokenizer_partitions_v{VERSION}.pkl", "wb") as partitions,\
            open(f"{path}BPETokenizer_merge_rules_v{VERSION}.pkl", "wb") as merge_rules:
            
            # Only need these two, can reuse corpus to continue training
            pickle.dump(bpe.word_partitions, partitions)
            pickle.dump(bpe.merge_rules, merge_rules)



bpe = BPE(evidence.loc[:,"processed evidence"], 15000, min_count=4)
bpe.train()

