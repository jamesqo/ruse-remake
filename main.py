from os import path
from typing import Iterator, List, Dict
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.dataset import Subset

from allennlp.common.file_utils import cached_path
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField, TextField, SequenceLabelField
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.elmo import Elmo, ELMoTokenCharactersIndexer
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import Covariance, CategoricalAccuracy, PearsonCorrelation
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

from embedders import ELMoTextFieldEmbedder

torch.manual_seed(1)

class WmtDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": ELMoTokenCharactersIndexer()}

    def text_to_instance(self,
                         mt_tokens: List[Token],
                         ref_tokens: List[Token],
                         human_score: float,
                         origin: str) -> Instance:
        mt_sent = TextField(mt_tokens, self.token_indexers)
        ref_sent = TextField(ref_tokens, self.token_indexers)
        human_score = ArrayField(np.array([human_score]))
        origin = MetadataField(origin)

        return Instance({"mt_sent": mt_sent,
                         "ref_sent": ref_sent,
                         "human_score": human_score,
                         "origin": origin})

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, mode='r', encoding='utf-8') as file:
            for line in file:
                mt_text, ref_text, score_text, origin = line.strip().split('\t')
                mt_words, ref_words, human_score = mt_text.split(), ref_text.split(), float(score_text)
                yield self.text_to_instance(
                        [Token(word) for word in mt_words],
                        [Token(word) for word in ref_words],
                        human_score,
                        origin)

class RuseModel(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder

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

    def forward(self,
                mt_sent: Dict[str, torch.Tensor],
                ref_sent: Dict[str, torch.Tensor],
                human_score: np.ndarray) -> Dict[str, torch.Tensor]:
        mt_mask = get_text_field_mask(mt_sent)
        ref_mask = get_text_field_mask(ref_sent)

        mt_embeddings = self.word_embeddings(mt_sent)
        ref_embeddings = self.word_embeddings(ref_sent)

        mt_encoder_out = self.encoder(mt_embeddings, mt_mask)
        ref_encoder_out = self.encoder(ref_embeddings, ref_mask)
    
        input = torch.cat((mt_encoder_out,
                           ref_encoder_out,
                           torch.mul(mt_encoder_out, ref_encoder_out),
                           torch.abs(mt_encoder_out - ref_encoder_out)), 1)
        reg = self.mlp(input)
        output = {'reg': reg}

        if human_score is not None:
            # run metric calculation
            self.covar(reg, human_score)
            self.pearson(reg, human_score)

            # calculate mean squared error
            delta = reg - human_score
            output['loss'] = torch.mul(delta, delta).sum()

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"covar": self.covar.get_metric(reset),
                "pearson": self.pearson.get_metric(reset)}

def select_origin(dataset, origin):
    indices = []
    for i, instance in enumerate(dataset):
        org = instance.fields['origin'].metadata
        if org == origin:
            indices.append(i)
    return indices

THIS_DIR = path.dirname(path.realpath(__file__))
DATA_DIR = path.join(THIS_DIR, 'data', 'trg-en')
DATASET_PATH = path.join(DATA_DIR, 'combined')
OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
WEIGHTS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

reader = WmtDatasetReader()
dataset = reader.read(cached_path(DATASET_PATH))

wmt2015_dataset = Subset(dataset, select_origin(dataset, 'newstest2015'))
wmt2016_dataset = Subset(dataset, select_origin(dataset, 'newstest2016'))
wmt2017_dataset = Subset(dataset, select_origin(dataset, 'newstest2017'))

vocab = Vocabulary.from_instances(dataset)
# TODO: Figure out the best parameters here
elmo = Elmo(cached_path(OPTIONS_FILE),
            cached_path(WEIGHTS_FILE),
            num_output_representations=2,
            dropout=0)
word_embeddings = ELMoTextFieldEmbedder({"tokens": elmo})
lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(input_size=elmo.get_output_dim(),
                                           hidden_size=64,
                                           num_layers=2,
                                           batch_first=True))

model = RuseModel(word_embeddings, lstm, vocab)
optimizer = optim.SGD(model.parameters(), lr=0.1)
iterator = BucketIterator(batch_size=2, sorting_keys=[("mt_sent", "num_tokens")])
iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  cuda_device=0,
                  train_dataset=(wmt2015_dataset + wmt2016_dataset),
                  validation_dataset=wmt2017_dataset,
                  patience=10,
                  num_epochs=1000)
trainer.train()
