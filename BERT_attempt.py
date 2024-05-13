import time
from multiprocessing.pool import ThreadPool
import pickle
import logging
from datetime import datetime

# Logging setup
LOGGING = True
if LOGGING:
    log_filename = datetime.now().strftime("%m-%d_%H-%M") + "_logfile.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO)

with open("evidence_v1.pkl","rb") as f:

    evidence = pickle.load(f)
    if LOGGING:
        logging.info("evidence loaded")


# Barely speeds up running time if at all???
# Thank GIL :c definitely not I/O bound
N_THREADS = 2


# Surprisingly couldn't find an implementation on torch/tf/keras
# Byte Pair Encoding tokenizer to feed into BERT-like model below
# Some sections referenced from 
# https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt
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
        self.merges = {}


    def train(self):

        t_start = time.time()
        for text in self.corpus:
            i = 0
            for word in text:
                self.word_freq[word] = self.word_freq.get(word, 0) + 1
                self.word_freq_partitions[i % N_THREADS][word] = self.word_freq_partitions[i % N_THREADS].get(word, 0)+1
        
        alphabet = set(("</w>",))
        for word in self.word_freq:

            # initialize word partitions
            char_list = list(word)
            char_list.append("</w>")
            self.word_partitions[word] = char_list

            # construct alphabet
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
        # 05/13: Nope (no cuda on local machine :c)
        iter = 0
        while len(self.vocab) < self.vocab_size:

            #t_s = time.time()
            # ~4s singlethread
            pair_freq = self.get_pair_freq()

            #print(f"pair_freq: {time.time()-t_s:.2f}s")
            

            if len(pair_freq) == 0 or max(pair_freq.values()) < self.min_count:
                print(f"Time {time.time()-t_start:.2f} - No more pairs. Exiting at vocab size {len(self.vocab)}")

            
            # Gets the pair that appears most in the training set
            # TODO: Optionally implement WordPiece critera:
            #       i.e. max( freq(left+right)/ (freq(left)*freq(right)) )
            best_pair = max(pair_freq, key=pair_freq.get)

            #t_s = time.time()
            self.update_word_partitions(best_pair[0], best_pair[1])
            
            # ~3s singlethread
            #print(f"update_word_partitions: {time.time()-t_s:.2f}s")

            # practically instant
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.append(best_pair[0] + best_pair[1])


            # Sanity keeping every 100 iterations - whole loop takes HOURS
            iter += 1
            if iter % 100 == 0:
                print(f"Time {time.time()-t_start:.2f} - Now on iteration {iter}.")
                if LOGGING:
                    logging.info(f"Iteration {iter} completed.")


    # Returns a pair frequency dictionary with entries:
    # pair (tuple) : freq (int)
    def get_pair_freq(self):
        pair_freq = {}

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
            
            return

        # returns None, results are already aggregated by the time all threads join.
        pool.map(get_pair_freq_subtask, self.word_freq_partitions)


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
                    partition[i] = partition[i] + partition[i+1]
                    
                    del partition[i+1]
                    length -= 1 # Otherwise runs out of bounds
    
                i += 1




    
    def tokenize(self, s):
            word_partitions = [list(t) + ["</w>"] for t in s.word_partition()]
    
            for left, right in self.merges:
                for index, word_partition in enumerate(word_partitions):
                    new_word_partition = []
                    i = 0
                    while i < len(word_partition):
                        if (
                            i + 1 < len(word_partition)
                            and word_partition[i] == left
                            and word_partition[i + 1] == right
                        ):
                            new_word_partition.append(left + right)
                            i += 2
                        else:
                            new_word_partition.append(word_partition[i])
                            i += 1
                    assert "".join(new_word_partition) == "".join(word_partition)
                    word_partitions[index] = new_word_partition
    
            return sum(word_partitions, [])
    

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(bpe, f)


bpe = BPE(evidence.loc[:,"processed evidence"], 20000, min_count=4)
bpe.train()

# Save to a model to be reused later
with open("BPETokenizer_v1", "wb") as f:
    pickle.dump(bpe, f)