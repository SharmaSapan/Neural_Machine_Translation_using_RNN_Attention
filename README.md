# Neural_Machine_Translation_using_RNN_Attention
Neural Machine Translation with Recurrent Neural Networks using attention mechanism. Evaluating this architecture by comparing its translation capability using two different recurrent cell types, namely Long Short-term Memory (LSTMs) and Gated Recurrent Units(GRUs), when given short sequence of words, versus when given a long sequence of words.


### Problem Defination
Performance of the Recurrent Neural
Network using Attention mechanism, two different sets of
data will be evaluated upon using two different recurrent
cell types, namely Long Short-term Memory
(LSTMs) and Gated Recurrent Units(GRUs). One
data set is generally smaller in each of its data
sample word sequence size. While another data set
will include data samples with larger sequences of
words. This can be interpreted as a difficult task for
humans to accomplish when trying to translate from
one language to another in an efficient manner, as
longer sequences may add more difficulty in translation
due to added complexity in word sequence
structure meanings.
The datasets were trimmed down to meet the
computation requirements of the system. The following
specification were used to train the models:

A. Portuguese to English Data Set (Long Sequence
Data) 
* There is a total of 40,000 different data samples
* Split into 3 different portions with 10/75/15
ratio:
  * Testing Portion (4,000 Data Samples)
  * Training Portion (30,000 Data Samples)
  * Validating Portion (6,000 Data Samples)
* All data samples are in plain text format
* All data sample text is in lowercase, and include basic punctuation
* Example of a data sample:
  * Portuguese Text: de volta `a minha pergunta
: porque ´e que fiquei ?
  * English Text: back to my question : why
did i stay ?

B. Portuguese to English Data Set (Short Sequence
Data)
* There is a total of 50,000 different data samples
* Split into 3 different portions with 10/75/15
ratio:
  * Testing Portion (5,000 Data Samples)
  * Training Portion (37,500 Data Samples)
  * Validating Portion (7500 Data Samples)
* All data samples are in plain text format
* All data sample text is in lowercase, and include
complete punctuation
* Example of a data sample:
  * Portuguese Text: Corre!
  * English Text: Run!

Each model will be assessed against each data set
mentioned above using BLEU scores to determine
how model performed after training. The model will
be assessed to determine how accurately the task of
translating languages can be completed, using the
appropriate testing data portions in each data set.

Recurrent Neural Networks (RNNs) are used to
learn how to map between an input sequence to
an output sequence. In general, most versions of
the Neural Machine Translation approach are made
up of 2 main parts; the Encoder, and the Decoder,
where both uses RNNs.

Architecture of Encoder-Decoder with attention

<img src="https://user-images.githubusercontent.com/54603828/133147683-0dddc837-0e6c-41c5-835b-54da38edce84.png" width="600" height="300" />

The encoder uses the word embeddings generated from tokenized words
indexed by the embedding layer as input
and passed through Recurrent layer which is made
up of LSTM or GRU in separate training models.
The LSTM cells provides 3 gates and is more
complex, where as the GRU cells has only two gates
and maintains only one state.

The decoder component in the NMT will assess
the input sequence s by each vector, and compile
target vectors one by one for the entire vector s [1].
This breakdown from the decoder will result in a
probability condition of:

<img src="https://latex.codecogs.com/gif.latex?log&space;P(B|A)&space;=&space;\sum_{j=1}^{m}&space;log&space;P(b_j|b_<j,s)" /></a>

decoder uses a context vector
from attention layer which is trained over time-steps
to calculate attention vectors, which is the concept
behind understanding where to place attention on
the words from the sentence to generate a prediction
probability.

Attention layer compares every source states
from encoder to the current target hidden state
in the decoder. A score matrix is generated using
Bahdanau’s additive style.
scores(h_t; h_s) = v tanh(W_1*h_t +W_2*h_t)
These scores are passed through a softmax function
to find the attention weights. A context vector
ct which is the weighted average of source states
and alignment score is combined with current target
hidden state h_t from decoder to get final attention
vector. Prediction after passing through the softmax function will be used as input to the next
time step in the decoder as well as to calculate the
loss and hidden state will be preserved.

Results were observed using various methods,
such as directly inferring result with state-of-the-art
methods like google translate, using BLEU score
to check the n-gram scores at all orders from 1 to
n and weighting them by calculation the weighted
geometric mean. Attention plots were generated
which is a way to easily visualize the alignment
matrix between the source and target sentence to
understand which words are getting more attention.
It was observed that the layers with the GRU
units performed better than the layers with LSTM
units on both long and short sequence of data.
The models were able to perform better on short
sequence of sentences compared to long sequence
of sentences as was expected from the limitation of
RNN based attention mechanism.

It had a similar score
as the accuracy metric of 0.17 for LSTM and 0.18
for GRU with short sequence. Surprisingly it had
higher score for long sequences as more words were
matching due to longer length. LSTM with long
sequences had 0.189 and GRU had 0.194 as average
BLEU scores on long sequences
Long sequences were trained under same architecture
and conditions but still had a high loss after
training of 10 epochs and taking considerably longer
time. The batch loss metric was started at 5.085
for LSTM and reached 1.822 after training, with
accuracy metric of 15. Similar results of 5.201 to
1.621 was observed on GRU using long sequences.
Results from individual translations were observed
and found that long sequences were translated
but lost the context mid sentence. Short sentences
like ”How long?” generated ”quanto tempo
?” which means ’How much time’ which was close
but had better context. Sentences like ”please come
back?” generated ”por favor , volte .” which has
exact meaning and word structure as the original
translation. One particularly interesting was the ”por
que nao va embora ?” which was translated from
”Why not go away?” and had a attention plot
attending the same words as seen in the figure

<img src="https://user-images.githubusercontent.com/54603828/133155503-6ab09567-8308-4a4e-9922-791bf8b5bc14.png" width="300" height="300" />

Long sequence of sentences such as ”see, some
people might fear girls.” tested on long sequence
trained model using GRU had a translation of ”veem
, as pessoas que os seus olhos .” which as seen from
Google translate is ”see, the people that your eyes.”
which is closer to the original but lost context mid
sentence but the attention plot as seen in the figure
6 is paying attention to the corresponding words.

<img src="https://user-images.githubusercontent.com/54603828/133159142-bcafba4a-3c2f-4bbf-b38c-eb4d0d484518.png" width="300" height="300" />

There are some limitations of this method when dealing with long sequence of data such as exploding gradient, vanishing gradient
problem, memory not long enough for long term
dependencies, no parallelization potential. A better and improved method which improves on these limitations is using breakthorugh Transformer model for Neural Machine Translation, which is explored in another repository.
