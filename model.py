import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import EmbeddingLayer


class TargetAwareBehaviorContextEncoder(nn.Module):
    """Candidate-aware attention pooling over one behavior history."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.user_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.context_norm = nn.LayerNorm(d_model)
        self.user_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        history_embed: torch.Tensor,
        history_ids: torch.Tensor,
        candidate_embed: torch.Tensor,
    ) -> torch.Tensor:
        # history_embed: [B, L, D], candidate_embed: [B, C, D]
        q = self.query(candidate_embed)
        k = self.key(history_embed)
        v = self.value(history_embed)

        logits = torch.einsum("bcd,bld->bcl", q, k) / math.sqrt(q.size(-1))
        valid_mask = history_ids.ne(0)
        logits = logits.masked_fill(~valid_mask.unsqueeze(1), -1e4)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        attn = attn * valid_mask.unsqueeze(1).float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        context = torch.einsum("bcl,bld->bcd", attn, v)
        return self.context_norm(context)

    def target_user(self, target_context: torch.Tensor) -> torch.Tensor:
        return self.user_norm(self.user_proj(target_context))


class BehaviorModalReliabilityScorer(nn.Module):
    """Sample-wise reliability scorer for a behavior-modality pair."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 5, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        aux_ctx: torch.Tensor,
        target_ctx: torch.Tensor,
        modal_prior: torch.Tensor,
        behavior_emb: torch.Tensor,
        modality_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([aux_ctx, target_ctx, modal_prior, behavior_emb, modality_emb], dim=-1)
        return torch.sigmoid(self.mlp(x))


class StableNoisyDecomposer(nn.Module):
    """Context-aware stable/noisy decomposition for text and image priors."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        def stable_branch():
            return nn.Sequential(
                nn.Linear(d_model * 4, d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
            )

        def noisy_branch():
            return nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.text_stable = stable_branch()
        self.image_stable = stable_branch()
        self.text_noisy = noisy_branch()
        self.image_noisy = noisy_branch()

    def forward(
        self,
        z_text: torch.Tensor,
        z_img: torch.Tensor,
        aux_ctx: torch.Tensor,
        target_ctx: torch.Tensor,
        behavior_emb: torch.Tensor,
    ):
        text_input = torch.cat([z_text, aux_ctx, target_ctx, behavior_emb], dim=-1)
        image_input = torch.cat([z_img, aux_ctx, target_ctx, behavior_emb], dim=-1)
        s_text = self.text_stable(text_input)
        s_img = self.image_stable(image_input)
        n_text = self.text_noisy(z_text)
        n_img = self.image_noisy(z_img)
        return s_text, n_text, s_img, n_img


class HEM3BSR(nn.Module):
    """Target-Oriented Stable Modal Transfer model.

    The class name is kept for compatibility with the original training script.
    """

    SUPPORTED_ABLATIONS = {
        "none",
        "ours",
        "base_only",
        "direct_fusion",
        "naive_fusion",
        "no_reliability",
        "wo_rel",
        "global_reliability",
        "global_rel",
        "no_decompose",
        "wo_snd",
        "no_align",
        "no_sparse",
    }

    ABLATION_ALIASES = {
        "ours": "none",
        "wo_rel": "no_reliability",
        "global_rel": "global_reliability",
        "wo_snd": "no_decompose",
        "naive_fusion": "direct_fusion",
    }

    def __init__(
        self,
        num_items: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        diffusion_timesteps: int = 1000,
        text_embeddings_path: str = None,
        image_embeddings_path: str = None,
        lambda_t: float = 0.1,
        lambda_orth: float = 0.01,
        lambda_align: float = 0.01,
        lambda_sparse: float = 0.001,
        ablation: str = "none",
    ):
        super().__init__()
        if ablation not in self.SUPPORTED_ABLATIONS:
            raise ValueError(f"Unsupported ablation={ablation!r}. Expected one of {sorted(self.SUPPORTED_ABLATIONS)}")

        self.num_items = num_items
        self.d_model = d_model
        self.diffusion_timesteps = diffusion_timesteps
        self.lambda_t = lambda_t
        self.lambda_orth = lambda_orth
        self.lambda_align = lambda_align
        self.lambda_sparse = lambda_sparse
        self.ablation = self.ABLATION_ALIASES.get(ablation, ablation)

        self.embedding_layer = EmbeddingLayer(
            num_items=num_items,
            d_model=d_model,
            text_embeddings_path=text_embeddings_path,
            image_embeddings_path=image_embeddings_path,
        )
        self.context_encoder = TargetAwareBehaviorContextEncoder(d_model, dropout)
        self.behavior_embedding = nn.Embedding(2, d_model)
        self.modality_embedding = nn.Embedding(2, d_model)
        self.reliability = BehaviorModalReliabilityScorer(d_model, dropout)
        self.decomposer = StableNoisyDecomposer(d_model, dropout)
        self.global_modal_logits = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        click_id_seq: torch.Tensor,
        click_img_seq: torch.Tensor,
        click_txt_seq: torch.Tensor,
        favor_id_seq: torch.Tensor,
        favor_img_seq: torch.Tensor,
        favor_txt_seq: torch.Tensor,
        labels: torch.Tensor = None,
        candidate_indices: torch.Tensor = None,
        return_loss: bool = True,
    ):
        device = click_id_seq.device
        batch_size = click_id_seq.size(0)

        click_id_embed, _, _ = self.embedding_layer(click_id_seq, click_img_seq, click_txt_seq)
        favor_id_embed, _, _ = self.embedding_layer(favor_id_seq, favor_img_seq, favor_txt_seq)

        candidates, candidate_mode = self._prepare_candidates(batch_size, candidate_indices, device)
        logits, aux_losses = self._score_candidates(
            click_id_embed=click_id_embed,
            click_ids=click_id_seq,
            favor_id_embed=favor_id_embed,
            favor_ids=favor_id_seq,
            candidates=candidates,
        )

        if not return_loss:
            return logits

        target = self._build_targets(labels, candidate_mode, logits.size(0), logits.size(1), device)
        rec_loss = self._recommendation_loss(logits, target, candidate_mode)
        total_loss = rec_loss + self._weighted_auxiliary_loss(aux_losses)
        return total_loss, logits

    def _prepare_candidates(self, batch_size: int, candidate_indices: torch.Tensor, device: torch.device):
        if candidate_indices is not None:
            return candidate_indices.to(device), True
        all_items = torch.arange(self.num_items, dtype=torch.long, device=device)
        return all_items.unsqueeze(0).expand(batch_size, -1), False

    def _score_candidates(
        self,
        click_id_embed: torch.Tensor,
        click_ids: torch.Tensor,
        favor_id_embed: torch.Tensor,
        favor_ids: torch.Tensor,
        candidates: torch.Tensor,
    ):
        batch_size, num_candidates = candidates.shape
        candidate_item_embed = self.embedding_layer.id_proj(self.embedding_layer.item_embedding(candidates))

        aux_ctx = self.context_encoder(click_id_embed, click_ids, candidate_item_embed)
        target_ctx = self.context_encoder(favor_id_embed, favor_ids, candidate_item_embed)
        user_target_ctx = self.context_encoder.target_user(target_ctx)

        flat_candidates = candidates.reshape(-1)
        flat_item_embed = candidate_item_embed.reshape(-1, self.d_model)
        flat_z_text = self.embedding_layer.lookup_text(flat_candidates)
        flat_z_img = self.embedding_layer.lookup_image(flat_candidates)
        flat_aux_ctx = aux_ctx.reshape(-1, self.d_model)
        flat_target_ctx = target_ctx.reshape(-1, self.d_model)
        flat_user_ctx = user_target_ctx.reshape(-1, self.d_model)

        enhanced_item_embed, aux_losses = self._enhance_items(
            flat_aux_ctx=flat_aux_ctx,
            flat_target_ctx=flat_target_ctx,
            flat_item_embed=flat_item_embed,
            flat_z_text=flat_z_text,
            flat_z_img=flat_z_img,
        )

        flat_user_ctx = F.normalize(flat_user_ctx, p=2, dim=-1)
        enhanced_item_embed = F.normalize(enhanced_item_embed, p=2, dim=-1)
        logits = torch.sum(flat_user_ctx * enhanced_item_embed, dim=-1).view(batch_size, num_candidates)
        return logits, aux_losses

    def _enhance_items(
        self,
        flat_aux_ctx: torch.Tensor,
        flat_target_ctx: torch.Tensor,
        flat_item_embed: torch.Tensor,
        flat_z_text: torch.Tensor,
        flat_z_img: torch.Tensor,
    ):
        zero = flat_item_embed.new_tensor(0.0)
        aux_losses = {"orth": zero, "align": zero, "sparse": zero}

        if self.ablation == "base_only":
            return flat_item_embed, aux_losses

        if self.ablation == "direct_fusion":
            enhanced_item_embed = flat_item_embed + self.lambda_t * (flat_z_text + flat_z_img)
            return enhanced_item_embed, aux_losses

        behavior_emb = self._aux_behavior_embedding(flat_item_embed.size(0), flat_item_embed.device)

        if self.ablation == "no_decompose":
            s_text, n_text = flat_z_text, torch.zeros_like(flat_z_text)
            s_img, n_img = flat_z_img, torch.zeros_like(flat_z_img)
            use_decompose_losses = False
        else:
            s_text, n_text, s_img, n_img = self.decomposer(
                flat_z_text,
                flat_z_img,
                flat_aux_ctx,
                flat_target_ctx,
                behavior_emb,
            )
            use_decompose_losses = True

        r_text, r_img = self._compute_reliability(
            flat_aux_ctx,
            flat_target_ctx,
            flat_z_text,
            flat_z_img,
            behavior_emb,
        )

        transfer = r_text * s_text + r_img * s_img
        enhanced_item_embed = flat_item_embed + self.lambda_t * transfer

        if use_decompose_losses:
            aux_losses["orth"] = self._orth_loss(s_text, n_text) + self._orth_loss(s_img, n_img)
            if self.ablation != "no_align":
                aux_losses["align"] = self._align_loss(s_text, flat_item_embed) + self._align_loss(s_img, flat_item_embed)
        if self.ablation not in {"no_sparse", "no_reliability"}:
            aux_losses["sparse"] = r_text.mean() + r_img.mean()

        return enhanced_item_embed, aux_losses

    def _aux_behavior_embedding(self, n: int, device: torch.device) -> torch.Tensor:
        # Current datasets use click/view as auxiliary behavior and favor/buy as target behavior.
        behavior_ids = torch.zeros(n, dtype=torch.long, device=device)
        return self.behavior_embedding(behavior_ids)

    def _compute_reliability(
        self,
        flat_aux_ctx: torch.Tensor,
        flat_target_ctx: torch.Tensor,
        flat_z_text: torch.Tensor,
        flat_z_img: torch.Tensor,
        behavior_emb: torch.Tensor,
    ):
        if self.ablation == "no_reliability":
            fixed = flat_z_text.new_full((flat_z_text.size(0), 1), 0.5)
            return fixed, fixed

        if self.ablation == "global_reliability":
            weights = torch.softmax(self.global_modal_logits, dim=0)
            r_text = weights[0].expand(flat_z_text.size(0), 1)
            r_img = weights[1].expand(flat_z_img.size(0), 1)
            return r_text, r_img

        n = flat_z_text.size(0)
        text_mod_ids = torch.zeros(n, dtype=torch.long, device=flat_z_text.device)
        img_mod_ids = torch.ones(n, dtype=torch.long, device=flat_z_text.device)
        text_mod_emb = self.modality_embedding(text_mod_ids)
        img_mod_emb = self.modality_embedding(img_mod_ids)

        r_text = self.reliability(flat_aux_ctx, flat_target_ctx, flat_z_text, behavior_emb, text_mod_emb)
        r_img = self.reliability(flat_aux_ctx, flat_target_ctx, flat_z_img, behavior_emb, img_mod_emb)
        return r_text, r_img

    def _build_targets(
        self,
        labels: torch.Tensor,
        candidate_mode: bool,
        batch_size: int,
        num_candidates: int,
        device: torch.device,
    ) -> torch.Tensor:
        if labels is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        labels = labels.to(device).long().view(-1)
        if candidate_mode:
            if labels.numel() > 0 and labels.max().item() < num_candidates:
                return labels
            return torch.zeros_like(labels)
        return labels

    def _recommendation_loss(self, logits: torch.Tensor, target: torch.Tensor, candidate_mode: bool) -> torch.Tensor:
        if not candidate_mode or logits.size(1) <= 1:
            return F.cross_entropy(logits, target)

        pos_score = logits.gather(1, target.view(-1, 1))
        neg_mask = torch.ones_like(logits, dtype=torch.bool)
        neg_mask.scatter_(1, target.view(-1, 1), False)
        neg_scores = logits[neg_mask].view(logits.size(0), -1)
        return -F.logsigmoid(pos_score - neg_scores).mean()

    def _weighted_auxiliary_loss(self, aux_losses: dict) -> torch.Tensor:
        return (
            self.lambda_orth * aux_losses["orth"]
            + self.lambda_align * aux_losses["align"]
            + self.lambda_sparse * aux_losses["sparse"]
        )

    @staticmethod
    def _orth_loss(stable: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        stable = F.normalize(stable, p=2, dim=-1)
        noisy = F.normalize(noisy, p=2, dim=-1)
        return torch.mean(torch.sum(stable * noisy, dim=-1) ** 2)

    @staticmethod
    def _align_loss(stable: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        stable = F.normalize(stable, p=2, dim=-1)
        target = F.normalize(target.detach(), p=2, dim=-1)
        return torch.mean(1.0 - torch.sum(stable * target, dim=-1))
