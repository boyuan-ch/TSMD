import torch.nn as nn
from models.model import TripleSumm


class TripleSummModeCls(TripleSumm):
    """TripleSumm with an auxiliary "behavior-mode" classification head tapped
    at the very beginning of the network, before any temporal or cross-modal
    processing.

    Unlike the genre head (`TripleSummGenreV1`, tapped after the MST blocks
    of the last model layer) this head reads the mean-pooled fusion stream
    immediately after modality projection + LayerNorm + averaging — i.e.
    before positional encoding, modality embeddings, and all MST/CMF blocks.
    The hypothesis (see `ana_mode.ipynb`) is that priming the very first
    representation with curve-shape awareness (the video's `gt_score`
    behavior cluster, discovered by data-driven clustering rather than the
    hand-crafted 7-type rule tree in `ana_dataset.ipynb`) helps the backbone
    allocate its limited capacity better than a late-tapped signal does.

    Returns: (frame_scores, mode_logits, attn_weights_list)
    """

    def __init__(
        self,
        visual_dim, text_dim, audio_dim, input_dim, hidden_dim,
        num_model_layers, num_mst_layers, num_cmf_layers,
        num_heads, dropout, window_size, max_seq_len, get_attn_weights,
        num_genre_classes=10,
    ):
        super().__init__(
            visual_dim=visual_dim, text_dim=text_dim, audio_dim=audio_dim,
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_model_layers=num_model_layers, num_mst_layers=num_mst_layers,
            num_cmf_layers=num_cmf_layers, num_heads=num_heads, dropout=dropout,
            window_size=window_size, max_seq_len=max_seq_len,
            get_attn_weights=get_attn_weights,
        )
        self.mode_head = nn.Linear(input_dim, num_genre_classes)
        self.mode_head.apply(self._init_weights)

    def forward(self, visual, text, audio, mask=None):
        # --- Projection & normalisation ---
        visual = self.visual_proj(visual)
        text   = self.text_proj(text)
        audio  = self.audio_proj(audio)

        visual = self.visual_ln(visual)
        text   = self.text_ln(text)
        audio  = self.audio_ln(audio)

        fusion = (visual + text + audio) / 3

        # --- Mode tap: right after fusion, before PE / modality embeddings / any blocks ---
        if mask is not None:
            denom = mask.float().sum(1, keepdim=True).clamp(min=1)
            mode_feat = (fusion * mask.unsqueeze(-1).float()).sum(1) / denom
        else:
            mode_feat = fusion.mean(1)
        mode_logits = self.mode_head(mode_feat)

        # --- Positional encoding & modality embeddings ---
        fusion = self.temporal_pe(fusion)
        visual = self.temporal_pe(visual)
        text   = self.temporal_pe(text)
        audio  = self.temporal_pe(audio)

        fusion = fusion + self.modality_embedding(fusion, modality_index=0)
        visual = visual + self.modality_embedding(visual, modality_index=1)
        text   = text   + self.modality_embedding(text,   modality_index=2)
        audio  = audio  + self.modality_embedding(audio,  modality_index=3)

        # --- Stacked model layers ---
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

        # --- Scoring head ---
        if mask is not None:
            fusion = fusion * mask.unsqueeze(-1).float()

        out = self.head(fusion).squeeze(-1)
        return out, mode_logits, attn_weights_list
