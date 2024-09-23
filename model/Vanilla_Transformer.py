import torch.nn as nn
from .positional_embedding import PositionalEmbedding
import math
from .dropout import Dropout
from .transformer_decoder_layer import TransformerDecoderLayer
from .transformer_encoder_layer import TransformerEncoderLayer
import torch

class TransformerModel(nn.Module):
    def __init__(self, args, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args
        self.supports_align_args = True



class TransformerEncoder(nn.Module):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__()
        self.dictionary = dictionary

        self.dropout_module = Dropout(args.dropout)
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = math.sqrt(embed_dim)

        self.embed_positions = PositionalEmbedding(
            embed_dim,
            self.padding_idx,
            init_size= args.max_source_positions + self.padding_idx + 1,
        )
        self.layernorm_embedding = None

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        self.layer_norm = None

    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def forward_embedding(
        self, src_tokens, token_embedding= None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens = False,
        token_embeddings = None,
    ):

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        # T x B x C, B x T, B x T x C, List[T x B x C]
        return x,  encoder_padding_mask, encoder_embedding, encoder_states
        
    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)



class TransformerDecoder(nn.Module):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__()
        self.args = args
        self.dictionary = dictionary
        self._future_mask = torch.empty(0)

        self.dropout_module = Dropout(args.dropout)
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = math.sqrt(embed_dim)

        self.project_in_dim = nn.Linear(input_embed_dim, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.project_in_dim.weight)

        self.embed_positions = PositionalEmbedding(
            embed_dim,
            self.padding_idx,
            init_size= args.max_source_positions + self.padding_idx + 1,
        )


        self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        self.layer_norm = None

        self.project_out_dim = nn.Linear(self.embed_dim, self.output_embed_dim, bias=False)
        nn.init.xavier_uniform_(self.project_out_dim.weight)

        self.adaptive_softmax = None
        self.output_projection = None

        if self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return TransformerDecoderLayer(args, no_encoder_attn)

    def forward(
        self,
        prev_output_tokens,
        encoder_out = None,
        incremental_state = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer = None,
        alignment_heads= None,
        src_lengths = None,
        return_all_hiddens = False,
    ):

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out = None,
        incremental_state= None,
        full_context_alignment: bool = False,
        alignment_layer = None,
        alignment_heads = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out = None,
        incremental_state = None,
        full_context_alignment: bool = False,
        alignment_layer = None,
        alignment_heads = None,
    ):
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn = None
        inner_states = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x, self._future_mask)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(tensor, future_mask):

        def fill_with_neg_inf(t):
            """FP16-compatible function that fills a tensor with -inf."""
            return t.float().fill_(float("-inf")).type_as(t)
        
        dim = tensor.size(0)
        if (
            future_mask.size(0) == 0
            or (not future_mask.device == tensor.device)
            or future_mask.size(0) < dim
        ):
            future_mask = torch.triu(
                fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        future_mask = future_mask.to(tensor)
        return future_mask[:dim, :dim]