#### In AllenNLP we use type annotations for just about everything.
from typing import Iterator, List, Dict

#### AllenNLP is built on top of PyTorch, so we use its code freely.
import torch
import torch.optim as optim
import numpy as np

#### In AllenNLP we represent each training example as an 
#### <code>Instance</code> containing <code>Field</code>s of various types. 
#### Here each example will have a <code>TextField</code> containing the sentence, 
#### and a <code>SequenceLabelField</code> containing the corresponding part-of-speech tags.
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField

#### Typically to solve a problem like this using AllenNLP, 
#### you'll have to implement two classes. The first is a 
#### <a href ="https://allenai.github.io/allennlp-docs/api/allennlp.data.dataset_readers.html">DatasetReader</a>, 
#### which contains the logic for reading a file of data and producing a stream of <code>Instance</code>s.
from allennlp.data.dataset_readers import DatasetReader

#### Frequently we'll want to load datasets or models from URLs. 
#### The <code>cached_path</code> helper downloads such files, 
#### caches them locally, and returns the local path. It also 
#### accepts local file paths (which it just returns as-is).
from allennlp.common.file_utils import cached_path

#### There are various ways to represent a word as one or more indices. 
#### For example, you might maintain a vocabulary of unique words and 
#### give each word a corresponding id. Or you might have one id per 
#### character in the word and represent each word as a sequence of ids. 
#### AllenNLP uses a has a <code>TokenIndexer</code> abstraction for this representation.
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

#### Whereas a <code>TokenIndexer</code> represents a rule for 
#### how to turn a token into indices, a <code>Vocabulary</code> 
#### contains the corresponding mappings from strings to integers. 
#### For example, your token indexer might specify to represent a 
#### token as a sequence of character ids, in which case the 
#### <code>Vocabulary</code> would contain the mapping {character -> id}. 
#### In this particular example we use a <code>SingleIdTokenIndexer</code> 
#### that assigns each token a unique id, and so the <code>Vocabulary</code> 
#### will just contain a mapping {token -> id} (as well as the reverse mapping).
from allennlp.data.vocabulary import Vocabulary

#### Besides <code>DatasetReader</code>, the other class you'll typically 
#### need to implement is <code>Model</code>, which is a PyTorch <code>Module</code> 
#### that takes tensor inputs and produces a dict of tensor outputs 
#### (including the training <code>loss</code> you want to optimize).
from allennlp.models import Model

#### As mentioned above, our model will consist of an embedding layer,
#### followed by a LSTM, then by a feedforward layer. AllenNLP includes 
#### abstractions for all of these that smartly handle padding and batching, 
#### as well as various utility functions.
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

#### We'll want to track accuracy on the training and validation datasets.
from allennlp.training.metrics import CategoricalAccuracy

#### In our training we'll need a <code>DataIterator</code>s that can intelligently batch our data.
from allennlp.data.iterators import BucketIterator

#### And we'll use AllenNLP's full-featured <code>Trainer</code>.
from allennlp.training.trainer import Trainer

#### Finally, we'll want to make predictions on new inputs, more about this below.
from allennlp.predictors import SentenceTaggerPredictor

torch.manual_seed(1)

#### Our first order of business is to implement our <code>DatasetReader</code> subclass.
class MyDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like
        The###DET dog###NN ate###V the###DET apple###NN
    """
    #### The only parameter our <code>DatasetReader</code> needs is a dict of 
    #### <code>TokenIndexer</code>s that specify how to convert tokens into indices. 
    #### By default we'll just generate a single index for each token (which we'll call "tokens") 
    #### that's just a unique id for each distinct token. (This is just the standard 
    #### "word to index" mapping you'd use in most NLP tasks.)
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    #### <code>DatasetReader.text_to_instance</code> takes the inputs corresponding
    #### to a training example (in this case the tokens of the sentence and the
    #### corresponding part-of-speech tags), instantiates the corresponding
    #### <a href="https://github.com/allenai/allennlp/blob/master/tutorials/notebooks/data_pipeline.ipynb"><code>Field</code>s</a>
    #### (in this case a <code>TextField</code> for the sentence and a <code>SequenceLabelField</code> 
    #### for its tags), and returns the <code>Instance</code> containing those fields.
    #### Notice that the tags are optional, since we'd like to be able to create instances 
    #### from unlabeled data to make predictions on them.
    def text_to_instance(self, mt_tokens: List[Token], ref_tokens: List[Token], human_score: float) -> Instance:
        mt_sentence_field = TextField(mt_tokens, self.token_indexers)
        ref_sentence_field = TextField(ref_tokens, self.token_indexers)
        fields = {"mt_sentence": mt_sentence_field, "ref_sentence": ref_sentence_field, "human_score": human_score}

        return Instance(fields)

    #### The other piece we have to implement is <code>_read</code>, 
    #### which takes a filename and produces a stream of <code>Instance</code>s.
    #### Most of the work has already been done in <code>text_to_instance</code>.
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                mt_sent, ref_sent, score_str = line.strip().split('\t')
                mt_words, ref_words, human_score = mt_sent.split(), ref_sent.split(), float(score_str)
                yield self.text_to_instance(map(Token, mt_words), map(Token, ref_words), human_score)

def main():
    thisdir = os.path.dirname(os.path.realpath(__file__))
    reader = MyDatasetReader()

    dataset = reader.read(cached_path(
        os.path.combine(filedir, 'data', 'combined')))

if __name__ == '__main__':
    main()
