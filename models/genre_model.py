import torch.nn as nn
from models.model import TripleSumm


class TripleSummGenreV1(TripleSumm):
    """TripleSumm with an auxiliary genre classification head.

    After the MST blocks of the last model layer (before the final CMF blocks),
    the fusion stream is mean-pooled over valid (unmasked) frames and forwarded
    through a linear classifier with `num_genre_classes` outputs.

    Returns: (frame_scores, genre_logits, attn_weights_list)
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
        self.genre_head = nn.Linear(input_dim, num_genre_classes)
        self.genre_head.apply(self._init_weights)

    def forward(self, visual, text, audio, mask=None):
        # --- Projection & normalisation ---
        visual = self.visual_proj(visual)
        text   = self.text_proj(text)
        audio  = self.audio_proj(audio)

        visual = self.visual_ln(visual)
        text   = self.text_ln(text)
        audio  = self.audio_ln(audio)

        fusion = (visual + text + audio) / 3

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
        genre_logits = None
        attn_weights_list = []
        for i in range(self.num_model_layers):
            # Multi-Scale Temporal blocks
            for j in range(self.num_mst_layers):
                fusion, visual, text, audio = self.temporal_block[
                    i * self.num_mst_layers + j
                ](fusion, visual, text, audio, mask)

            # Genre tap: after the MST blocks of the last model layer
            if i == self.num_model_layers - 1:
                if mask is not None:
                    denom = mask.float().sum(1, keepdim=True).clamp(min=1)
                    genre_feat = (fusion * mask.unsqueeze(-1).float()).sum(1) / denom
                else:
                    genre_feat = fusion.mean(1)
                genre_logits = self.genre_head(genre_feat)

            # Cross-Modal Fusion blocks
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
        return out, genre_logits, attn_weights_list


class TripleSummGenreV2(TripleSummGenreV1):
    """TripleSummGenreV1 with a deeper MLP genre head.

    Genre head: Linear(input_dim, 16) -> GELU -> Dropout(0.2) -> LayerNorm(16) -> Linear(16, num_genre_classes)
    Note: no Sigmoid — CrossEntropyLoss expects raw logits.
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
            num_genre_classes=num_genre_classes,
        )
        # Replace the simple linear head with a deeper MLP
        self.genre_head = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(16),
            nn.Linear(16, num_genre_classes),
        )
        self.genre_head.apply(self._init_weights)


# V3 is architecturally identical to V2; the difference is the loss weight
# decay schedule defined in configs/mosu-genrev3.yaml and applied in solver.py.
TripleSummGenreV3 = TripleSummGenreV2
