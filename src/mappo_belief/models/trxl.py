import torch
import torch.nn as nn


def _build_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError("Unsupported TrXL activation: {}".format(name))


class BuiltinTrXLEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_heads,
        n_layers,
        dropout=0.0,
        activation="relu",
        norm_first=False,
        final_norm=False,
        ff_mult=4,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if final_norm else None
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, norm=encoder_norm)

    def forward(self, x):
        return self.encoder(x)


class GRUGatingUnit(nn.Module):
    """GTrXL-style residual gate between the residual stream x and update y."""

    def __init__(self, hidden_dim, bias_init=2.0):
        super().__init__()
        self.w_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.u_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.u_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_g = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.u_g = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bg = nn.Parameter(torch.full((hidden_dim,), bias_init))

    def forward(self, x, y):
        r = torch.sigmoid(self.w_r(y) + self.u_r(x))
        z = torch.sigmoid(self.w_z(y) + self.u_z(x) - self.bg)
        h = torch.tanh(self.w_g(y) + self.u_g(r * x))
        return (1.0 - z) * x + z * h


class CustomTrXLBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_heads,
        dropout=0.0,
        activation="relu",
        norm_first=False,
        ff_mult=4,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Keep the self_written path aligned with the qmix GTrXL-style block:
        # normalize before each sub-layer, then fuse updates with GRU gates.
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_gate = GRUGatingUnit(hidden_dim)
        self.ff_gate = GRUGatingUnit(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            _build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )

    def forward(self, x):
        attn_in = self.attn_norm(x)
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in, need_weights=False)
        x = self.attn_gate(x, self.dropout(attn_out))

        ff_in = self.ff_norm(x)
        ff_out = self.dropout(self.ff(ff_in))
        x = self.ff_gate(x, ff_out)
        return x


class CustomTrXLEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_heads,
        n_layers,
        dropout=0.0,
        activation="relu",
        norm_first=False,
        final_norm=False,
        ff_mult=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CustomTrXLBlock(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                    ff_mult=ff_mult,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim) if final_norm else None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x


def build_trxl_encoder(
    hidden_dim,
    n_heads,
    n_layers,
    dropout=0.0,
    activation="relu",
    norm_first=False,
    final_norm=False,
    ff_mult=4,
    impl="builtin",
):
    impl_name = str(impl).lower()
    if impl_name == "builtin":
        return BuiltinTrXLEncoder(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            final_norm=final_norm,
            ff_mult=ff_mult,
        )
    if impl_name in ("custom", "self_written", "manual"):
        return CustomTrXLEncoder(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            final_norm=final_norm,
            ff_mult=ff_mult,
        )
    raise ValueError("Unsupported trxl_impl: {}".format(impl))
