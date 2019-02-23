import os
from os import path

from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import ArrayField, TextField, SequenceLabelField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics import Covariance, CategoricalAccuracy, PearsonCorrelation

from allennlp.data.iterators import BucketIterator

from allennlp.training.trainer import Trainer

#from allennlp.predictors import SentenceTaggerPredictor

torch.manual_seed(1)

class MyDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like
        The###DET dog###NN ate###V the###DET apple###NN
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, mt_tokens: List[Token], ref_tokens: List[Token], human_score: float) -> Instance:
        mt_sent_field = TextField(mt_tokens, self.token_indexers)
        ref_sent_field = TextField(ref_tokens, self.token_indexers)
        human_score_field = ArrayField(np.array([human_score]))
        fields = {"mt_sent": mt_sent_field, "ref_sent": ref_sent_field, "human_score": human_score_field}

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                mt_sent, ref_sent, score_str = line.strip().split("\t")
                mt_words, ref_words, human_score = mt_sent.split(), ref_sent.split(), float(score_str)
                yield self.text_to_instance(
                        [Token(word) for word in mt_words],
                        [Token(word) for word in ref_words],
                        human_score)

class Ruse(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder, encoder: Seq2VecEncoder, vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        #self.mlp = torch.nn.Linear(in_features=encoder.get_output_dim(),
        #          out_features=1)
        hidden_dim = 128
        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(in_features=encoder.get_output_dim()*4, out_features=hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(in_features=hidden_dim, out_features=1)
            )
        self.covar = Covariance()
        self.pearson = PearsonCorrelation()

    def forward(self, mt_sent: Dict[str, torch.Tensor], ref_sent: Dict[str, torch.Tensor],
            human_score: np.ndarray) -> Dict[str, torch.Tensor]:
        mt_mask, ref_mask = get_text_field_mask(mt_sent), get_text_field_mask(ref_sent)
        mt_embeddings, ref_embeddings = self.word_embeddings(mt_sent), self.word_embeddings(ref_sent)
        mt_enc_out, ref_enc_out = self.encoder(mt_embeddings, mt_mask), self.encoder(ref_embeddings, ref_mask)
    
        cat = torch.cat((mt_enc_out, ref_enc_out, torch.mul(mt_enc_out, ref_enc_out), torch.abs(mt_enc_out - ref_enc_out)), 1)
        reg = self.mlp(cat)
        output = {'reg' : reg}
        #print('lr', lr)
        #print('human', human_score)

        if human_score is not None:
            # running metric calculation
            self.covar(reg, human_score)
            self.pearson(reg, human_score)

            # calculate mean squared error
            delta = reg - human_score
            output['loss'] = torch.mul(delta, delta).sum()

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"covar": self.covar.get_metric(reset), 'pearson': self.pearson.get_metric(reset)}


#reader = PosDatasetReader()
#train_dataset = reader.read(cached_path(
#    'https://raw.githubusercontent.com/allenai/allennlp'
#    '/master/tutorials/tagger/training.txt'))
#validation_dataset = reader.read(cached_path(
#    'https://raw.githubusercontent.com/allenai/allennlp'
#    '/master/tutorials/tagger/validation.txt'))
thisdir = path.dirname(path.realpath(__file__))
datadir = path.join(thisdir, "data", "trg-en")
reader = MyDatasetReader()

dataset = reader.read(cached_path(
    path.join(datadir, "combined")))

vocab = Vocabulary.from_instances(dataset)

EMBEDDING_DIM = 64
HIDDEN_DIM = 64
NUM_LAYERS = 2

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True))

model = Ruse(word_embeddings, lstm, vocab)

optimizer = optim.SGD(model.parameters(), lr=0.1)

iterator = BucketIterator(batch_size=2, sorting_keys=[("mt_sent", "num_tokens")])
iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  cuda_device=0,
                  train_dataset=dataset,
                  validation_dataset=dataset,
                  patience=10,
                  num_epochs=1000)

trainer.train()

#predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
#tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
#tag_ids = np.argmax(tag_logits, axis=-1)
#print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

# Here's how to save the model.
with open("/tmp/model.th", 'wb') as f:
    torch.save(model.state_dict(), f)
vocab.save_to_files("/tmp/vocabulary")

# And here's how to reload the model.
vocab2 = Vocabulary.from_files("/tmp/vocabulary")
model2 = LstmTagger(word_embeddings, lstm, vocab2)
with open("/tmp/model.th", 'rb') as f:
    model2.load_state_dict(torch.load(f))

#predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
#tag_logits2 = predictor2.predict("The dog ate the apple")['tag_logits']
#assert tag_logits2 == tag_logits
