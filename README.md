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
<img src="https://user-images.githubusercontent.com/54603828/133147683-0dddc837-0e6c-41c5-835b-54da38edce84.png)" width="300" height="300" />

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
- <img src="https://latex.codecogs.com/gif.latex?log P(B | A )=\sigma \text { Probability of a sensor reading value when sleep onset is observed at a time bin } t " />
and like the encoder, the Recurrent Neural Network
can be leveraged to perform this decoder
compilation.




