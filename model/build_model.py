from .models import XTransformerEncoder, XTransformerDecoder, XTransformerModel
import torch
import torch.nn as nn

def build_model(args, dicts):
    langs = [lang for lang in args.langs]

    embed_tokens = {}
    for lang in langs:
        if len(embed_tokens) == 0 or args.share_all_embeddings is False:
            embed_token = build_embedding(
                dicts[lang], args.encoder_embed_dim, args.encoder_embed_path,
            )
            embed_tokens[lang] = embed_token
        else:
            embed_tokens[lang] = embed_tokens[langs[0]]

    args.share_decoder_input_output_embed = True
    encoders, decoders = {}, {}

    for lang in langs:
        encoder_embed_tokens = embed_tokens[lang]
        decoder_embed_tokens = encoder_embed_tokens
        if lang in args.source_langs:
            encoder = XTransformerEncoder(args, dicts[lang], encoder_embed_tokens)
            encoders[lang] = encoder
        if lang in args.target_langs:
            decoder = XTransformerDecoder(args, dicts[lang], decoder_embed_tokens)
            decoders[lang] = decoder
    return XTransformerModel(encoders, decoders, args.eval_lang_pair)



def build_embedding(dictionary, embed_dim, path=None):
    num_embeddings = len(dictionary) # 21904
    padding_idx = dictionary.pad()
    emb = Embedding(num_embeddings, embed_dim, padding_idx) 
    # if provided, load from preloaded dictionaries
    if path:
        embed_dict = parse_embedding(path)
        load_embedding(embed_dict, dictionary, emb)
    return emb


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def parse_embedding(embed_path):
    embed_dict = {}
    with open(embed_path) as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor(
                [float(weight) for weight in pieces[1:]]
            )
    return embed_dict


def load_embedding(embed_dict, vocab, embedding):
    for idx in range(len(vocab)):
        token = vocab[idx]
        if token in embed_dict:
            embedding.weight.data[idx] = embed_dict[token]
    return embedding