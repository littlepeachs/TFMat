import torch
import torch.nn as nn


class FrozenTextEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        output_dim: int,
        pooling: str = 'mean',
        proj_hidden_dim: int | None = None,
        dropout: float = 0.0,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for FrozenTextEncoder. Install with `pip install -e .[text]`."
            ) from exc

        self.encoder = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self.pooling = pooling
        self.freeze_encoder = freeze_encoder
        hidden_size = self.encoder.config.hidden_size
        self.n_out = output_dim

        if proj_hidden_dim is not None and proj_hidden_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, proj_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(proj_hidden_dim, output_dim),
            )
        elif output_dim != hidden_size:
            self.proj = nn.Linear(hidden_size, output_dim)
        else:
            self.proj = nn.Identity()

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        masked_hidden = hidden_states * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        return masked_hidden.sum(dim=1) / denom

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == 'cls':
            return hidden_states[:, 0]
        if self.pooling == 'mean':
            return self._mean_pool(hidden_states, attention_mask)
        raise ValueError(f"Unknown text pooling mode: {self.pooling}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        return self.proj(pooled)