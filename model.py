from typing import Dict

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Covariance, PearsonCorrelation

import numpy as np
import torch

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
                human_score: np.ndarray,
                origin: str) -> Dict[str, torch.Tensor]:
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
        output = {"reg": reg}

        # TODO: Under what conditions does this occur?
        if human_score is not None:
            # run metric calculation
            self.covar(reg, human_score)
            self.pearson(reg, human_score)

            # calculate mean squared error
            delta = reg - human_score
            output["loss"] = torch.mul(delta, delta).sum()

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"covar": self.covar.get_metric(reset),
                "pearson": self.pearson.get_metric(reset)}
