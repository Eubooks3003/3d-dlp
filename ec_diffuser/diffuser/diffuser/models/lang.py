"""
CLIPTextEncoder
---------------

Frozen CLIP text encoder + cache. Produces per-instruction token sequences that
AdaLNPINTDenoiser consumes as additional particles.

Two-stage split:
  - CLIPTextEncoder.encode(strings) runs once per unique instruction (pre-train
    or policy-reset), returns a CPU tensor of shape [B, L_lang, clip_dim].
  - LangProjection (a Linear layer inside the denoiser) maps clip_dim ->
    projection_dim at each forward pass.

Why frozen + cached: CLIP embeddings don't change during diffusion training;
encoding once per episode avoids paying for the 63M-param text transformer on
every denoising step.

Default model: openai/clip-vit-base-patch32 (ViT-B/32), clip_dim=512, L_lang=77.
"""
import torch
from torch import nn


CLIP_DIM = 512
CLIP_MAX_LEN = 77


class CLIPTextEncoder(nn.Module):
    """
    Frozen CLIP text encoder. Not a torch Module in the trainable sense --
    parameters are frozen and not registered for optimizer. Kept as nn.Module
    only so it can be .to(device)'d.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cpu", return_pooled: bool = False):
        super().__init__()
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError as e:
            raise ImportError(
                "CLIPTextEncoder requires `transformers`. Install with: "
                "pip install transformers"
            ) from e

        self.model_name = model_name
        self.return_pooled = return_pooled
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name).to(device)
        for p in self.text_model.parameters():
            p.requires_grad = False
        self.text_model.eval()
        self._device = device

    @property
    def clip_dim(self) -> int:
        return self.text_model.config.hidden_size

    @torch.no_grad()
    def encode(self, strings):
        """
        Args:
            strings: list[str] of length B (or a single str)
        Returns:
            If return_pooled: (B, clip_dim) EOS-pooled embedding.
            Else: (B, L, clip_dim) last_hidden_state, attention_mask (B, L).
        """
        if isinstance(strings, str):
            strings = [strings]
        tok = self.tokenizer(
            strings, padding="max_length", truncation=True,
            max_length=CLIP_MAX_LEN, return_tensors="pt",
        ).to(self._device)
        out = self.text_model(**tok)
        if self.return_pooled:
            return out.pooler_output  # (B, clip_dim)
        return out.last_hidden_state, tok["attention_mask"]  # (B, L, D), (B, L)


class LanguageInstructionCache:
    """
    Caches CLIP embeddings keyed by the exact instruction string. Useful because
    RLBench episodes share instructions across variations and paraphrases repeat
    across tasks -- we encode each unique string once.

    Storage format (pooled=False):
        self._cache[string] = (embedding: (L, clip_dim), mask: (L,))
    """

    def __init__(self, encoder: CLIPTextEncoder):
        self.encoder = encoder
        self._cache = {}

    def get(self, string: str):
        if string in self._cache:
            return self._cache[string]
        if self.encoder.return_pooled:
            emb = self.encoder.encode([string])[0].cpu()  # (clip_dim,)
            self._cache[string] = emb
            return emb
        emb, mask = self.encoder.encode([string])
        emb = emb[0].cpu()       # (L, clip_dim)
        mask = mask[0].cpu()     # (L,)
        self._cache[string] = (emb, mask)
        return (emb, mask)

    def get_batch(self, strings):
        """Convenience for a list. Returns stacked (B, L, D), (B, L)."""
        if self.encoder.return_pooled:
            embs = torch.stack([self.get(s) for s in strings], dim=0)
            return embs
        embs, masks = [], []
        for s in strings:
            e, m = self.get(s)
            embs.append(e)
            masks.append(m)
        return torch.stack(embs, dim=0), torch.stack(masks, dim=0)
