from .datasets import MusicMtDataset, MusicMassDataset, IndexedDataset, MMapIndexedDataset, RoundRobinZipDatasets,index_file_path, data_file_path
import itertools
import os
from collections import OrderedDict

def _get_mass_dataset_key(lang_pair):
    return "mass:" + lang_pair

def _get_mt_dataset_key(lang_pair):
    return "" + lang_pair


def infer_dataset_impl(path):
    if IndexedDataset.exists(path):
        with open(index_file_path(path), "rb") as f:
            magic = f.read(8)
            if magic == IndexedDataset._HDR_MAGIC:
                return "cached"
            elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
                return "mmap"
            else:
                return None
    else:
        return None
    
def make_dataset(path, impl, fix_lua_indexing=False, dictionary=None):
    if impl == "lazy" and IndexedDataset.exists(path):
        return IndexedDataset(path, fix_lua_indexing=fix_lua_indexing)
    elif impl == "mmap" and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path)
    return None

def load_indexed_dataset(
    path, dictionary=None, dataset_impl=None, combine=False, default="cached"
):

    datasets = []
    for k in itertools.count():
        path_k = path + (str(k) if k > 0 else "")

        dataset_impl_k = dataset_impl
        if dataset_impl_k is None:
            dataset_impl_k = infer_dataset_impl(path_k)
        dataset = make_dataset(
            path_k,
            impl=dataset_impl_k or default,
            fix_lua_indexing=True,
            dictionary=dictionary,
        )
        if dataset is None:
            break
        datasets.append(dataset)
        if not combine:
            break
    if len(datasets) == 0:
        return None
    elif len(datasets) == 1:
        return datasets[0]
    else:
        return None


def build_datasets(args, dicts, split, training):

    def split_exists(split, lang):
        filename = os.path.join(args.data, '{}.{}'.format(split, lang))
        return os.path.exists(index_file_path(filename)) and os.path.exists(data_file_path(filename))

    def split_para_exists(split, key, lang):
        filename = os.path.join(args.data, '{}.{}.{}'.format(split, key, lang))
        return os.path.exists(index_file_path(filename)) and os.path.exists(data_file_path(filename))
    
    src_mono_datasets = {}
    for lang_pair in args.mono_lang_pairs: #lyric-lyric, melody-melody
        lang = lang_pair.split('-')[0]
        if split_exists(split, lang):
            prefix = os.path.join(args.data, '{}.{}'.format(split, lang))
        else:
            raise FileNotFoundError('Not Found available {} dataset for ({}) lang'.format(split, lang))

        src_mono_datasets[lang_pair] = load_indexed_dataset(prefix, dicts[lang])
        print('| monolingual {}-{}: {} examples'.format(split, lang, len(src_mono_datasets[lang_pair])))

    src_para_datasets = {}
    for lang_pair in args.para_lang_pairs: # lyric-melody
        src, tgt = lang_pair.split('-')
        key = '-'.join(sorted([src, tgt]))
        if not split_para_exists(split, key, src):
            raise FileNotFoundError('Not Found available {}-{} para dataset for ({}) lang'.format(split, key, src))
        if not split_para_exists(split, key, tgt):
            raise FileNotFoundError('Not Found available {}-{} para dataset for ({}) lang'.format(split, key, tgt))

        prefix = os.path.join(args.data, '{}.{}'.format(split, key))
        if '{}.{}'.format(key, src) not in src_para_datasets:
            src_para_datasets[key + '.' + src] = load_indexed_dataset(prefix + '.' + src, dicts[src])
        if '{}.{}'.format(key, tgt) not in src_para_datasets:
            src_para_datasets[key + '.' + tgt] = load_indexed_dataset(prefix + '.' + tgt, dicts[tgt])

        print('| bilingual {} {}-{}.{}: {} examples'.format(
            split, src, tgt, src, len(src_para_datasets[key + '.' + src])
        ))
        print('| bilingual {} {}-{}.{}: {} examples'.format(
            split, src, tgt, tgt, len(src_para_datasets[key + '.' + tgt])
        ))

    mt_para_dataset = {}
    for lang_pair in args.mt_steps: # lyric-melody, melody-lyric
        src, tgt = lang_pair.split('-')
        key = '-'.join(sorted([src, tgt]))
        src_key = key + '.' + src
        tgt_key = key + '.' + tgt
        src_dataset = src_para_datasets[src_key]
        tgt_dataset = src_para_datasets[tgt_key]
        src_id, tgt_id = args.langs_id[src], args.langs_id[tgt]

        mt_para_dataset[lang_pair] = MusicMtDataset(
            src_dataset, src_dataset.sizes,
            tgt_dataset, tgt_dataset.sizes,
            dicts[src], dicts[tgt],
            src_id, tgt_id,
            left_pad_source=args.left_pad_source,
            left_pad_target=args.left_pad_target,
            max_source_positions=args.max_source_positions,
            max_target_positions=args.max_target_positions,
            src_lang=src,
            tgt_lang=tgt,
        )

    eval_para_dataset = {}
    if split != 'train':
        for lang_pair in args.valid_lang_pairs: #lyric-lyric, melody-melody
            src, tgt = lang_pair.split('-')
            src_id, tgt_id = args.langs_id[src], args.langs_id[tgt]
            if src == tgt:
                src_key = src + '-' + tgt
                tgt_key = src + '-' + tgt
                src_dataset = src_mono_datasets[src_key]
                tgt_dataset = src_mono_datasets[tgt_key]
            else:
                key = '-'.join(sorted([src, tgt]))
                src_key = key + '.' + src
                tgt_key = key + '.' + tgt
                src_dataset = src_para_datasets[src_key]
                tgt_dataset = src_para_datasets[tgt_key]
            eval_para_dataset[lang_pair] = MusicMtDataset(
                src_dataset, src_dataset.sizes,
                tgt_dataset, tgt_dataset.sizes,
                dicts[src], dicts[tgt],
                src_id, tgt_id,
                left_pad_source=args.left_pad_source,
                left_pad_target=args.left_pad_target,
                max_source_positions=args.max_source_positions,
                max_target_positions=args.max_target_positions,
                src_lang=src,
                tgt_lang=tgt,
            )

    mass_mono_datasets = {}
    if split == 'train':
        for lang_pair in args.mass_steps:
            src_dataset = src_mono_datasets[lang_pair]
            lang = lang_pair.split('-')[0]

            mass_mono_dataset = MusicMassDataset(
                src_dataset, src_dataset.sizes, dicts[lang],
                left_pad_source=args.left_pad_source,
                left_pad_target=args.left_pad_target,
                max_source_positions=args.max_source_positions,
                max_target_positions=args.max_target_positions,
                shuffle=True,
                lang_id=args.langs_id[lang],
                ratio=args.word_mask,
                pred_probs=args.pred_probs,
                lang=lang
            )
            mass_mono_datasets[lang_pair] = mass_mono_dataset
    return RoundRobinZipDatasets(
            OrderedDict([
                (_get_mt_dataset_key(lang_pair), mt_para_dataset[lang_pair])
                for lang_pair in mt_para_dataset.keys()
            ] + [
                (_get_mass_dataset_key(lang_pair), mass_mono_datasets[lang_pair])
                for lang_pair in mass_mono_datasets.keys()
            ] + [
                (_get_mt_dataset_key(lang_pair), eval_para_dataset[lang_pair])
                for lang_pair in eval_para_dataset.keys()
            ]),
            eval_key=None if training else args.eval_lang_pair
        )