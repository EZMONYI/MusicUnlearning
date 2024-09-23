from .XTransformer import XTransformerEncoder, XTransformerDecoder, XTransformerModel
import torch.nn as nn
from utils.dictionary import MaskedLMDictionary
import os
from collections import OrderedDict
from argparse import Namespace
import torch
from utils.dataload import build_datasets
from .embeddings import build_embedding


def build_songmass(ckpt_path):

    def build_model(args, dicts):
        langs = [lang for lang in args.langs]

        embed_tokens = {}
        for lang in langs:
            if len(embed_tokens) == 0 or args.share_all_embeddings is False:
                embed_token = build_embedding(dicts[lang], args.encoder_embed_dim)
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


    def prepare_args(args):
        args.left_pad_source = True
        args.left_pad_target = False
        s = args.word_mask_keep_rand.split(',')
        s = [float(x) for x in s]
        setattr(args, 'pred_probs', torch.FloatTensor([s[0], s[1], s[2]]))

        args.langs = sorted(args.langs.split(','))
        args.source_langs = sorted(args.source_langs.split(','))
        args.target_langs = sorted(args.target_langs.split(','))

        for lang in args.source_langs:
            assert lang in args.langs
        for lang in args.target_langs:
            assert lang in args.langs

        args.mass_steps = [s for s in args.mass_steps.split(',') if len(s) > 0]
        args.mt_steps = [s for s in args.mt_steps.split(',') if len(s) > 0]

        mono_langs = [
            lang_pair.split('-')[0]
            for lang_pair in args.mass_steps
            if len(lang_pair) > 0
        ]

        mono_lang_pairs = []
        for lang in mono_langs:
            mono_lang_pairs.append('{}-{}'.format(lang, lang))
        setattr(args, 'mono_lang_pairs', mono_lang_pairs)

        args.para_lang_pairs = list(set([
            '-'.join(sorted(lang_pair.split('-')))
            for lang_pair in set(args.mt_steps) if
            len(lang_pair) > 0
        ]))

        args.valid_lang_pairs = [s for s in args.valid_lang_pairs.split(',') if len(s) > 0]

        for lang_pair in args.mono_lang_pairs:
            src, tgt = lang_pair.split('-')
            assert src in args.source_langs and tgt in args.target_langs

        for lang_pair in args.valid_lang_pairs:
            src, tgt = lang_pair.split('-')
            assert src in args.source_langs and tgt in args.target_langs

        if args.source_lang is not None:
            assert args.source_lang in args.source_langs

        if args.target_lang is not None:
            assert args.target_lang in args.target_langs

        langs_id = {}
        ids_lang = {}
        for i, v in enumerate(args.langs):
            langs_id[v] = i
            ids_lang[i] = v
        setattr(args, 'langs_id', langs_id)
        setattr(args, 'ids_lang', ids_lang)

        # If provide source_lang and target_lang, we will switch to translation
        if args.source_lang is not None and args.target_lang is not None:
            setattr(args, 'eval_lang_pair', '{}-{}'.format(args.source_lang, args.target_lang))
            training = False
        else:
            if len(args.para_lang_pairs) > 0:
                required_para = [s for s in set(args.mt_steps)]
                setattr(args, 'eval_lang_pair', required_para[0])
            else:
                setattr(args, 'eval_lang_pair', args.mono_lang_pairs[0])
            training = True
        setattr(args, 'n_lang', len(langs_id))
        setattr(args, 'eval_para', True if len(args.para_lang_pairs) > 0 else False)
        return training


    args = Namespace(data='unlearn/processed', activation_dropout=0.1, max_source_positions=1024, encoder_layers=6, dropout=0.1, encoder_embed_dim=512, decoder_embed_dim=512, decoder_output_dim=512, encoder_normalize_before=False, decoder_normalize_before=False, share_decoder_input_output_embed=True, max_target_positions=1024, cross_self_attention=False, decoder_ffn_embed_dim=2048, share_all_embeddings=False,encoder_layerdrop=0,encoder_attention_heads=8, attention_dropout=0.1,encoder_ffn_embed_dim=2048, decoder_layers=6,decoder_attention_heads=8, langs = 'lyric,melody', source_lang=None,source_langs='lyric,melody', target_lang=None, target_langs='lyric,melody', valid_lang_pairs='lyric-lyric,melody-melody', mt_steps= 'lyric-melody,melody-lyric', mass_steps='lyric-lyric,melody-melody', word_mask_keep_rand='0.8,0.1,0.1', word_mask=0.25, ) 

    training = prepare_args(args)

    dicts = OrderedDict()

    for lang in args.langs:
        dicts[lang] = MaskedLMDictionary.load(os.path.join(args.unlearn_data, 'dict.{}.txt'.format(lang)))
        if len(dicts) > 0:
            assert dicts[lang].pad() == dicts[args.langs[0]].pad()
            assert dicts[lang].eos() == dicts[args.langs[0]].eos()
            assert dicts[lang].unk() == dicts[args.langs[0]].unk()
            assert dicts[lang].mask() == dicts[args.langs[0]].mask()
        print('| [{}] dictionary: {} types'.format(lang, len(dicts[lang])))

    model = build_model(args, dicts)
    

    unlearn_dataset = build_datasets(args, dicts, 'train', training)
    unlearn_dataset.ordered_indices()
    args.data='retrain/processed'
    retain_dataset = build_datasets(args, dicts, 'train', training)
    retain_dataset.ordered_indices()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    model_path = ckpt_path
    ckpt = torch.load(model_path)
    
    model.load_state_dict(ckpt, strict=False)

    # for name, param in model.named_parameters():
    #     print(name, param.data)

    return model, unlearn_dataset, retain_dataset