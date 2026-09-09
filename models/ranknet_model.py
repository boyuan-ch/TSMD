import torch
from models.model import TripleSumm


class TripleSummRankNet(TripleSumm):
    """
    TripleSumm variant that returns both pre-sigmoid logits and sigmoid scores.

    forward() returns (sigmoid_out, logit_out, attn_weights_list) instead of
    the base class (sigmoid_out, attn_weights_list), so that the training loop
    can pass pre-sigmoid logits to the RankNet loss while still using sigmoid
    scores for the MSE term.

    Architecture is identical to TripleSumm — only the forward signature differs.
    self.head is shared; the final Sigmoid layer (self.head[-1]) is simply
    bypassed to obtain logits: logit = self.head[:-1](fusion).squeeze(-1).
    """

    def forward(self, visual, text, audio, mask=None):
        visual = self.visual_proj(visual)
        text = self.text_proj(text)
        audio = self.audio_proj(audio)

        visual = self.visual_ln(visual)
        text = self.text_ln(text)
        audio = self.audio_ln(audio)

        fusion = (visual + text + audio) / 3

        fusion = self.temporal_pe(fusion)
        visual = self.temporal_pe(visual)
        text = self.temporal_pe(text)
        audio = self.temporal_pe(audio)

        fusion = fusion + self.modality_embedding(fusion, modality_index=0)
        visual = visual + self.modality_embedding(visual, modality_index=1)
        text = text + self.modality_embedding(text, modality_index=2)
        audio = audio + self.modality_embedding(audio, modality_index=3)

        attn_weights_list = []
        for i in range(self.num_model_layers):
            for j in range(self.num_mst_layers):
                fusion, visual, text, audio = self.temporal_block[i * self.num_mst_layers + j](fusion, visual, text, audio, mask)

            for j in range(self.num_cmf_layers):
                fusion, attn_weights = self.modality_block[i * self.num_cmf_layers + j](fusion, visual, text, audio)

            if self.get_attn_weights:
                attn_weights_list.append(attn_weights.detach())

        if mask is not None:
            fusion = fusion * mask.unsqueeze(-1).float()

        # Pre-sigmoid logits for ranking loss; sigmoid scores for MSE loss
        logits = self.head[:-1](fusion).squeeze(-1)   # (B, T_max)
        out = torch.sigmoid(logits)                    # (B, T_max)
        return out, logits, attn_weights_list
