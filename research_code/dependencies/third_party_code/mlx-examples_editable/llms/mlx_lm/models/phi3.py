# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .su_rope import SuScaledRotaryEmbedding


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: Optional[int] = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, List[float]]]] = None
    max_position_embeddings: int = 131072
    original_max_position_embeddings: int = 4096

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"long_factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] not in ["longrope", "su", "linear"]:
                print(
                    "[WARNING] rope_scaling 'type' currently only supports 'linear', 'su', and 'longrope'; setting rope scaling to false."
                )
                self.rope_scaling = None


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.num_hidden_layers = args.num_hidden_layers

        self.head_dim = head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        op_size = n_heads * head_dim + 2 * (n_kv_heads * head_dim)
        self.qkv_proj = nn.Linear(dim, op_size, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        if args.rope_scaling and args.rope_scaling["type"] in ["longrope", "su"]:
            self.rope = SuScaledRotaryEmbedding(
                head_dim,
                base=args.rope_theta,
                max_position_embeddings=args.max_position_embeddings,
                original_max_position_embeddings=args.original_max_position_embeddings,
                short_factor=args.rope_scaling["short_factor"],
                long_factor=args.rope_scaling["long_factor"],
            )
        else:
            rope_scale = 1.0
            if args.rope_scaling and args.rope_scaling["type"] == "linear":
                assert isinstance(args.rope_scaling["factor"], float)
                rope_scale = 1 / args.rope_scaling["factor"]
            self.rope = nn.RoPE(
                head_dim,
                traditional=args.rope_traditional,
                base=args.rope_theta,
                scale=rope_scale,
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.qkv_proj(x)
        query_pos = self.n_heads * self.head_dim
        queries, keys, values = mx.split(
            qkv, [query_pos, query_pos + self.n_kv_heads * self.head_dim], axis=-1
        )

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_up_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        x = self.gate_up_proj(x)
        gate, x = mx.split(x, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * x)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Phi3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = Phi3Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.args = args
        self.is_adapted = False
        self.distribution = 1
        self.is_v2 = True

    def get_mx_array_as_type_from_torch_file(self, adaptor_model_dir, filename, mx_type, is_conv=False):
        import torch
        from os import path
        if is_conv:
            return mx.array(torch.load(f"{path.join(adaptor_model_dir, filename)}", weights_only=True,
                                       map_location=torch.device("cpu")).numpy(), dtype=mx_type).transpose((0, 2, 1))
        else:
            return mx.array(torch.load(f"{path.join(adaptor_model_dir, filename)}", weights_only=True,
                       map_location=torch.device("cpu")).numpy(), dtype=mx_type)

    def add_adaptors(self, adaptor_model_dir, dual=False):
        self.fc_original = nn.Linear(self.args.hidden_size, self.args.vocab_size, bias=False)
        self.fc_negative = nn.Linear(self.args.hidden_size, self.args.vocab_size, bias=False)
        self.fc_positive = nn.Linear(self.args.hidden_size, self.args.vocab_size, bias=False)

        self.fc_original.weight = self.get_mx_array_as_type_from_torch_file(adaptor_model_dir,
                                                                            'ai_fc_original_weight.pt', mx.bfloat16)
        self.fc_negative.weight = self.get_mx_array_as_type_from_torch_file(adaptor_model_dir, 'ai_fc_negative_weight.pt', mx.bfloat16)

        self.fc_positive.weight = self.get_mx_array_as_type_from_torch_file(adaptor_model_dir, 'ai_fc_positive_weight.pt', mx.bfloat16)

        self.is_adapted = True
        self.distribution = 1
        self.dual = dual  # not used in this version


    def switch_distribution(self, distribution_id: int, merge_factor=1.0):
        if distribution_id == 0:
            print(f"Switching to negative distribution")
        if distribution_id == 1:
            print(f"Switching to positive distribution")
        if distribution_id == 3:
            print(f"Switching to joint positive + negative distribution (only for greedy decoding)")
        if distribution_id == -1:
            print(f"Switching to original distribution")
        self.distribution = distribution_id
        self.set_merge_factor(merge_factor=merge_factor)

    def set_merge_factor(self, merge_factor=1.0):
        self.merge_factor = merge_factor

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
        return_embeddings=False
    ):
        out = self.model(inputs, mask, cache)

        if self.is_adapted:
            if self.distribution == 0:
                return [self.fc_negative(out), out]
            elif self.distribution == 1:
                return [self.fc_positive(out), out]
            elif self.distribution == 3:
                # Not valid if sampling, but just for greedy decoding, this is equivalent to having the full
                # normalized distribution. The vocab is still the original vocab to match the existing inference code.
                # This is relatively inefficient, since we are taking the softmax twice (when combined with the
                # existing inference code). Note also that the sdm() activation is not used. And the probabilities are
                # not relevant.
                negative_distribution = self.fc_negative(out)
                positive_distribution = self.fc_positive(out)
                full_distribution = mx.concatenate([negative_distribution, positive_distribution], axis=-1)
                vocab_size = positive_distribution.shape[-1]
                full_distribution = mx.softmax(full_distribution, axis=-1)
                batch_argmax_indexes = mx.argmax(full_distribution, axis=-1) #% vocab_size  # mod delayed to below
                peaked_distribution = mx.zeros_like(positive_distribution) + 10e-12
                # Note the mod below
                peaked_distribution[:, mx.arange(positive_distribution.shape[1]), batch_argmax_indexes % vocab_size] = \
                    full_distribution[:, mx.arange(full_distribution.shape[1]), batch_argmax_indexes]
                return [mx.log(peaked_distribution), out]
            elif self.distribution == -1:
                return [self.lm_head(out), out]
            else:
                assert False
        else:
            return [self.lm_head(out), out]

    @property
    def layers(self):
        return self.model.layers
