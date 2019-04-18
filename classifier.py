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
                "accuracy" : CategoricalAccuracy(),
                "accuracy3" : CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()


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
        logits = self.mlp(input)
        probs = torch.nn.softmax(logits)
        output = {"probs": probs}

        if human_score is not None:
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
