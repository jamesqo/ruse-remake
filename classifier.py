from typing import Dict

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy

import torch

class InferSentModel(Model):
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
                torch.nn.Linear(in_features=hidden_dim, out_features=3)
            )

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self,
                sent1: Dict[str, torch.Tensor],
                sent2: Dict[str, torch.Tensor],
                label: str, # TODO: Is this the correct type?
                origin: str) -> Dict[str, torch.Tensor]:
        mask1, mask2 = map(get_text_field_mask, [sent1, sent2])
        embed1, embed2 = map(self.word_embeddings, [sent1, sent2])
        out1 = self.encoder(embed1, mask1)
        out2 = self.encoder(embed2, mask2)

        input = torch.cat((out1, out2, torch.mul(out1, out2), torch.abs(out1 - out2)), 1)
        logits = self.mlp(input)
        probs = torch.nn.softmax(logits)
        output = {"probs": probs}

        # XXX: Is this check still needed?
        # if human_score is not None:

        # run metric calculation
        for metric in self.metrics.values():
            metric(logits, label)

        # calculate loss
        loss = self.loss(logits, label)
        output['loss'] = loss

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.metrics['accuracy'].get_metric(reset),
                "accuracy3": self.metrics['accuracy3'].get_metric(reset)}
