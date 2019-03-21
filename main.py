from functools import partial
from os import path
import sys

from allennlp.common.file_utils import cached_path
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.elmo import Elmo
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper
from allennlp.training.trainer import Trainer

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.dataset import Subset

from embedders import ELMoTextFieldEmbedder
from grid_search import grid_search_iter
from kfold import StratifiedKFold
from model import RuseModel
from reader import WmtDatasetReader

torch.manual_seed(1)

def origin_of(instance):
    return instance.fields["origin"].metadata

def filter_by_origin(dataset, origin):
    indices = []
    for i, instance in enumerate(dataset):
        if origin_of(instance) == origin:
            indices.append(i)
    return Subset(dataset, indices)

def calculate_cv_loss(dataset, params):
    print("Using hyperparameter configuration:", params)

    vocab = Vocabulary.from_instances(dataset)
    # TODO: Figure out the best parameters here
    elmo = Elmo(cached_path(OPTIONS_FILE),
                cached_path(WEIGHTS_FILE),
                num_output_representations=2,
                dropout=params["dropout"]) # TODO: Does dropout refer to the LSTM or ELMo?
    word_embeddings = ELMoTextFieldEmbedder({"tokens": elmo})
    # TODO: Figure out the best parameters here
    lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(input_size=elmo.get_output_dim(),
                                               hidden_size=64,
                                               num_layers=params["num_layers"],
                                               batch_first=True))

    model = RuseModel(word_embeddings, lstm, vocab)
    optimizer = optim.Adam(model.parameters())
    # TODO: What kind of iterator should be used?
    iterator = BucketIterator(batch_size=params["batch_size"],
                              sorting_keys=[("mt_sent", "num_tokens"),
                                            ("ref_sent", "num_tokens")])
    iterator.index_with(vocab)

    losses = []
    kfold = StratifiedKFold(dataset, k=10, grouping=origin_of)
    for train, val in kfold:
        # TODO: Figure out best hyperparameters
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          cuda_device=0,
                          train_dataset=train,
                          validation_dataset=val,
                          patience=5,
                          num_epochs=100)
        trainer.train()
        # TODO: Better way to access the validation loss?
        loss, _ = trainer._validation_loss()
        losses.append(loss)
    mean_loss = np.mean(losses)
    print("Mean validation loss was:", mean_loss)
    return mean_loss

THIS_DIR = path.dirname(path.realpath(__file__))
DATA_DIR = path.join(THIS_DIR, 'data', 'trg-en')
DATASET_PATH = path.join(DATA_DIR, 'combined')
OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
WEIGHTS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

reader = WmtDatasetReader()
dataset = reader.read(cached_path(DATASET_PATH))

dataset15 = filter_by_origin(dataset, 'newstest2015')
dataset16 = filter_by_origin(dataset, 'newstest2016')
dataset17 = filter_by_origin(dataset, 'newstest2017')

grid = {
    "num_layers": [1, 2, 3],
    # TODO: num_units
    "batch_size": [64, 128, 256, 512, 1024],
    "dropout": [0.1, 0.3, 0.5]
}
all_params = grid_search_iter(grid)
# TODO: We should cache the results so we don't have to train again with these parameters
cv_loss_15_16 = partial(calculate_cv_loss, (dataset15 + dataset16))
best_params_15_16 = min(all_params, key=cv_loss_15_16)
print(best_params_15_16)
# TODO: Test on WMT17 dataset
