import torch.nn as nn
from models.model import TripleSumm


class TripleSummClsMSE(TripleSumm):
    """
    TripleSumm with a per-frame classification head and optional regression head.

    Adds a single linear layer (cls_head) that maps the per-frame fusion
    features to num_cls_bins class logits, applied at the same position
    as the regression head (after all transformer blocks, post-mask).

    gt_score in [0,1] is binned into num_cls_bins equal-width intervals:
      bin k covers [k/num_bins, (k+1)/num_bins), with score=1.0 clamped to bin k=num_bins-1.

        Returns: (regression_scores, cls_logits, attn_weights_list)
            regression_scores : (B, T_max) sigmoid scores, or None when disabled
      cls_logits        : (B, T_max, num_cls_bins) raw logits → focal / EMD loss
      attn_weights_list : list of attention weights (if get_attn_weights)
    """

    def __init__(
        self,
        visual_dim, text_dim, audio_dim, input_dim, hidden_dim,
        num_model_layers, num_mst_layers, num_cmf_layers,
        num_heads, dropout, window_size, max_seq_len, get_attn_weights,
        num_cls_bins=10,
        use_regression_head=True,
    ):
        super().__init__(
            visual_dim=visual_dim, text_dim=text_dim, audio_dim=audio_dim,
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_model_layers=num_model_layers, num_mst_layers=num_mst_layers,
            num_cmf_layers=num_cmf_layers, num_heads=num_heads, dropout=dropout,
            window_size=window_size, max_seq_len=max_seq_len,
            get_attn_weights=get_attn_weights,
        )
        # Single linear classification head — raw logits, no activation
        self.cls_head = nn.Linear(input_dim, num_cls_bins)
        self._init_weights(self.cls_head)
        if not use_regression_head:
            self.head = None

    def forward(self, visual, text, audio, mask=None):
        visual = self.visual_proj(visual)
        text   = self.text_proj(text)
        audio  = self.audio_proj(audio)

        visual = self.visual_ln(visual)
        text   = self.text_ln(text)
        audio  = self.audio_ln(audio)

        fusion = (visual + text + audio) / 3

        fusion = self.temporal_pe(fusion)
        visual = self.temporal_pe(visual)
        text   = self.temporal_pe(text)
        audio  = self.temporal_pe(audio)

        fusion = fusion + self.modality_embedding(fusion, modality_index=0)
        visual = visual + self.modality_embedding(visual, modality_index=1)
        text   = text   + self.modality_embedding(text,   modality_index=2)
        audio  = audio  + self.modality_embedding(audio,  modality_index=3)

        attn_weights_list = []
        for i in range(self.num_model_layers):
            for j in range(self.num_mst_layers):
                fusion, visual, text, audio = self.temporal_block[
                    i * self.num_mst_layers + j
                ](fusion, visual, text, audio, mask)
            for j in range(self.num_cmf_layers):
                fusion, attn_weights = self.modality_block[
                    i * self.num_cmf_layers + j
                ](fusion, visual, text, audio)
            if self.get_attn_weights:
                attn_weights_list.append(attn_weights.detach())

        if mask is not None:
            fusion = fusion * mask.unsqueeze(-1).float()

        out = None
        if self.head is not None:
            out = self.head(fusion).squeeze(-1)     # (B, T_max) sigmoid scores
        cls_logits = self.cls_head(fusion)          # (B, T_max, num_cls_bins)
        return out, cls_logits, attn_weights_list


class TripleSummSalienceCls(TripleSummClsMSE):
    """TripleSumm with only a per-frame salience classification head."""

    def __init__(self, *args, num_cls_bins=4, **kwargs):
        kwargs.pop('use_regression_head', None)
        super().__init__(
            *args,
            num_cls_bins=num_cls_bins,
            use_regression_head=False,
            **kwargs,
        )
