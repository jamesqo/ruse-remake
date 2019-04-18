from typing import Dict, Iterator, List

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField, TextField
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.modules.elmo import ELMoTokenCharactersIndexer

import numpy as np

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

class WmtClassifierDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": ELMoTokenCharactersIndexer()}

    def text_to_instance(self,
                         tokens_1: List[Token],
                         tokens_2: List[Token],
                         label: LabelField,
                         origin: str) -> Instance:
        sent_1 = TextField(tokens_1, self.token_indexers)
        sent_2 = TextField(tokens_1, self.token_indexers)
        origin = MetadataField(origin)

        return Instance({"sent_1": sent_1,
                         "sent_2": sent_2,
                         "label" : label,
                         "origin": origin})

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, mode='r', encoding='utf-8') as file:
            for line in file:
                mt_text, ref_text, score_text, origin = line.strip().split('\t')
                mt_words, ref_words, human_score = mt_text.split(), ref_text.split(), float(score_text)

                if mt_text != ref_text:
                    yield self.text_to_instance(
                            [Token(word) for word in mt_words],
                            [Token(word) for word in ref_words],
                            LabelField('left'),
                            origin)
 
                    yield self.text_to_instance(
                            [Token(word) for word in ref_words],
                            [Token(word) for word in mt_words],
                            LabelField('right'),
                            origin)
