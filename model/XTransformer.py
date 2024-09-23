import torch
import torch.nn as nn
import math
from .dropout import Dropout
from .positional_embedding import PositionalEmbedding
from .transformer_encoder_layer import TransformerEncoderLayer
from .transformer_decoder_layer import TransformerDecoderLayer, MaskedAttentionDecoderLayer
import torch.nn.functional as F



class XTransformerEncoder(nn.Module):
    def __init__(self, args, dictionary, embed_tokens) -> None:
        super().__init__()
        self.dictionary = dictionary
        self.mask_idx = dictionary.mask_index

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

    def forward(self, src_tokens, src_lengths, source_sent_ids=None, target_sent_ids=None):

        x = self.embed_scale * self.embed_tokens(src_tokens) # embedding
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens) # added sin positional embedding

        x = self.dropout_module(x)
        # B:Batch_len=7  T:padded_Token_len=158 C: embedding_len=512
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx) | src_tokens.eq(self.mask_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)
            
        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'source_sent_ids': source_sent_ids # B x S
        }
    
    
class XTransformerDecoder(nn.Module):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__()
        self.dictionary = dictionary

        self.args = args
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

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False)

        self.embed_positions = PositionalEmbedding(
            embed_dim,
            self.padding_idx,
            init_size= args.max_target_positions + self.padding_idx + 1,
        )


        self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.layer_norm = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False)

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
        
        self.layers = nn.ModuleList([])
        self.layers.extend([
            MaskedAttentionDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])
        self.num_layers = len(self.layers)
        self.cnt = 0

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return TransformerDecoderLayer(args, no_encoder_attn)
        
    def forward(self, prev_output_tokens, encoder_out=None,
                incremental_state=None, positions=None):
        if encoder_out is not None and type(encoder_out) == type(dict()) \
            and 'source_sent_ids' in encoder_out.keys() and encoder_out['source_sent_ids'] is not None:

            src_len = encoder_out['source_sent_ids'].size()[-1]
            tgt_len = prev_output_tokens.size()[1]
            beam_batch_size = prev_output_tokens.size()[0]

            source_sent_ids = encoder_out['source_sent_ids']
            is_sep = prev_output_tokens.eq(5).int()
            target_sent_ids = is_sep.cumsum(dim=1)
            
            # T is current time step
            s = source_sent_ids.unsqueeze(1).repeat(1, tgt_len, 1)
            t = target_sent_ids.unsqueeze(2).repeat(1, 1, src_len)
            sent_mask = torch.ne(s, t) 
            sent_mask = sent_mask[:, -1, :]
            sent_mask = sent_mask.unsqueeze(1)
            encoder_out['encoder_padding_mask'] = sent_mask

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
            positions=positions,
        ) if self.embed_positions is not None else None

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
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn, attns = None, []

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn, _ = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x, self._future_mask) if incremental_state is None else None,
                need_attn=True,
            )
            inner_states.append(x)
            attns.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states, 'attns': attns}

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

class XTransformerModel(nn.Module):
    def __init__(self, encoders, decoders, eval_lang_pair=None):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)
        self.tgt_key = None
        if eval_lang_pair is not None:
            self.source_lang = eval_lang_pair.split('-')[0]
            self.target_lang = eval_lang_pair.split('-')[1]

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        if torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)


    def max_decoder_positions(self):
        return min(decoder.max_positions() for decoder in self.decoders.values())

    def forward(self, src_tokens, src_lengths, prev_output_tokens,
                source_sent_ids, target_sent_ids, src_key, tgt_key, positions=None):

        encoder_out = self.encoders[src_key](src_tokens, src_lengths)

        input_encoder_out = encoder_out['encoder_out']
        input_encoder_padding_mask = encoder_out['encoder_padding_mask']

        src_len = src_tokens.size()[1]
        tgt_len = prev_output_tokens.size()[1]
        # (B, S) -> (B,1,S) -> (B,T,S)
        s = source_sent_ids.unsqueeze(1).repeat(1, tgt_len, 1)
        # (B, T) -> (B,T,1) -> (B,T,S)
        t = target_sent_ids.unsqueeze(2).repeat(1, 1, src_len)

        sent_mask = torch.ne(s, t)
        encoder_out['encoder_padding_mask'] = sent_mask

        decoder_out = self.decoders[tgt_key](
            prev_output_tokens,
            encoder_out=encoder_out,
            positions=positions
        )
        self.tgt_key = tgt_key
        return decoder_out

    @property
    def decoder(self):
        return self.decoders[self.target_lang]

    @property
    def encoder(self):
        return self.encoders[self.source_lang]




def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


