import numpy as np
import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """ID, text, and image embedding adapter used by TOSMT."""

    def __init__(
        self,
        num_items: int,
        d_model: int,
        text_embeddings_path: str = None,
        image_embeddings_path: str = None,
    ):
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model

        self.item_embedding = nn.Embedding(num_items, d_model)
        self.id_proj = nn.Linear(d_model, d_model)

        self._init_text_embedding(text_embeddings_path)
        self._init_image_embedding(image_embeddings_path)

    def _fit_embedding_rows(self, embeddings: torch.Tensor, name: str) -> torch.Tensor:
        """Align external embedding rows with item ids where 0 is padding."""
        if embeddings.dim() != 2:
            raise ValueError(f"{name} must be a 2D array, got shape={tuple(embeddings.shape)}")
        if embeddings.size(0) == self.num_items:
            return embeddings
        if embeddings.size(0) == self.num_items - 1:
            pad = embeddings.new_zeros(1, embeddings.size(1))
            return torch.cat([pad, embeddings], dim=0)
        if embeddings.size(0) < self.num_items:
            pad = embeddings.new_zeros(self.num_items - embeddings.size(0), embeddings.size(1))
            print(f"[Warn] {name} rows padded: {embeddings.size(0)} -> {self.num_items}")
            return torch.cat([embeddings, pad], dim=0)
        print(f"[Warn] {name} rows truncated: {embeddings.size(0)} -> {self.num_items}")
        return embeddings[: self.num_items]

    def _init_text_embedding(self, text_embeddings_path: str = None):
        if text_embeddings_path is None:
            self.text_embedding_mock = nn.Embedding(self.num_items, self.d_model)
            self.text_proj = nn.Linear(self.d_model, self.d_model)
            self.use_real_text = False
            return

        try:
            text_embeddings = torch.from_numpy(np.load(text_embeddings_path)).float()
            text_embeddings = self._fit_embedding_rows(text_embeddings, "text_embeddings")
            text_dim = text_embeddings.shape[1]
            self.text_proj = nn.Linear(text_dim, self.d_model) if text_dim != self.d_model else nn.Identity()
            self.text_embedding = nn.Parameter(text_embeddings, requires_grad=False)
            self.use_real_text = True
            print(f"[Info] Loaded text embeddings: {tuple(text_embeddings.shape)}")
        except Exception as exc:
            print(f"[Warn] Failed to load text embeddings, using mock embeddings: {exc}")
            self.text_embedding_mock = nn.Embedding(self.num_items, self.d_model)
            self.text_proj = nn.Linear(self.d_model, self.d_model)
            self.use_real_text = False

    def _init_image_embedding(self, image_embeddings_path: str = None):
        if image_embeddings_path is None:
            self.image_embedding_mock = nn.Embedding(self.num_items, self.d_model)
            self.img_proj = nn.Linear(self.d_model, self.d_model)
            self.use_real_image = False
            return

        try:
            image_embeddings = torch.from_numpy(np.load(image_embeddings_path)).float()
            image_embeddings = self._fit_embedding_rows(image_embeddings, "image_embeddings")
            img_dim = image_embeddings.shape[1]
            self.img_proj = nn.Linear(img_dim, self.d_model) if img_dim != self.d_model else nn.Identity()
            self.image_embedding = nn.Parameter(image_embeddings, requires_grad=False)
            self.use_real_image = True
            print(f"[Info] Loaded image embeddings: {tuple(image_embeddings.shape)}")
        except Exception as exc:
            print(f"[Warn] Failed to load image embeddings, using mock embeddings: {exc}")
            self.image_embedding_mock = nn.Embedding(self.num_items, self.d_model)
            self.img_proj = nn.Linear(self.d_model, self.d_model)
            self.use_real_image = False

    def lookup_text(self, item_ids: torch.Tensor) -> torch.Tensor:
        if self.use_real_text:
            return self.text_proj(self.text_embedding[item_ids])
        return self.text_proj(self.text_embedding_mock(item_ids))

    def lookup_image(self, item_ids: torch.Tensor) -> torch.Tensor:
        if self.use_real_image:
            return self.img_proj(self.image_embedding[item_ids])
        return self.img_proj(self.image_embedding_mock(item_ids))

    def forward(self, item_id_seq: torch.Tensor, image_seq: torch.Tensor, text_seq: torch.Tensor):
        id_embed = self.id_proj(self.item_embedding(item_id_seq))
        img_embed = self.lookup_image(image_seq)
        txt_embed = self.lookup_text(text_seq)
        return id_embed, img_embed, txt_embed
