import numpy as np
import torch
import copy
from functools import lru_cache
import struct
import os
from collections import OrderedDict


def index_file_path(prefix_path):
    return prefix_path + ".idx"

def data_file_path(prefix_path):
    return prefix_path + ".bin"

def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a

def _warmup_mmap_file(path):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass

def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.double,
    8: np.uint16,
}



class RoundRobinZipDatasets(torch.utils.data.Dataset):

    def __init__(self, datasets, eval_key=None):
        super().__init__()
        assert isinstance(datasets, OrderedDict)
        self.datasets = datasets
        self.eval_key = eval_key

        self.longest_dataset = None
        self.longest_dataset_key = None
        for key, dataset in datasets.items():
            assert isinstance(dataset, torch.utils.data.Dataset)
            if self.longest_dataset is None or len(dataset) > len(self.longest_dataset):
                self.longest_dataset = dataset
                self.longest_dataset_key = key

        self._ordered_indices = None

    def _map_index(self, key, index):
        assert (
            self._ordered_indices is not None
        ), "Must call RoundRobinZipDatasets.ordered_indices() first"
        return self._ordered_indices[key][index % len(self.datasets[key])]

    def __getitem__(self, index):
        if self.eval_key is None:
            return OrderedDict(
                [
                    (key, dataset[self._map_index(key, index)])
                    for key, dataset in self.datasets.items()
                ]
            )
        else:
            # at evaluation time it's useful to pass-through batches from a single key
            return self.datasets[self.eval_key][self._map_index(self.eval_key, index)]

    def __len__(self):
        return len(self.longest_dataset)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return None
        if self.eval_key is None:
            return OrderedDict(
                [
                    (key, dataset.collater([sample[key] for sample in samples]))
                    for key, dataset in self.datasets.items()
                ]
            )
        else:
            # at evaluation time it's useful to pass-through batches from a single key
            return self.datasets[self.eval_key].collater(samples)

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        # TODO make it configurable whether to use max() or sum() here
        return max(
            dataset.num_tokens(self._map_index(key, index))
            for key, dataset in self.datasets.items()
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return {
            key: dataset.size(self._map_index(key, index))
            for key, dataset in self.datasets.items()
        }

    def ordered_indices(self):
        """Ordered indices for batching."""
        if self._ordered_indices is None:
            # Call the underlying dataset's ordered_indices() here, so that we
            # get the same random ordering as we would have from using the
            # underlying dataset directly.
            self._ordered_indices = OrderedDict(
                [
                    (key, dataset.ordered_indices())
                    for key, dataset in self.datasets.items()
                ]
            )
        return np.arange(len(self))

    @property
    def supports_prefetch(self):
        return all(
            getattr(dataset, "supports_prefetch", False)
            for dataset in self.datasets.values()
        )

    def prefetch(self, indices):
        for key, dataset in self.datasets.items():
            dataset.prefetch([self._map_index(key, index) for index in indices])



class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, "wb")

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack("<Q", 1))
                    self._file.write(struct.pack("<B", code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack("<Q", len(sizes)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path):
        self._path = path
        self._index = self.Index(index_file_path(self._path))

        _warmup_mmap_file(data_file_path(self._path))
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        ptr, size = self._index[i]
        np_array = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr
        )
        if self._index.dtype != np.int64:
            np_array = np_array.astype(np.int64)

        return torch.from_numpy(np_array)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(
            data_file_path(path)
        )
    

class IndexedDataset(torch.utils.data.Dataset):
    """Loader for TorchNet IndexedDataset"""

    _HDR_MAGIC = b"TNTIDX\x00\x00"

    def __init__(self, path, fix_lua_indexing=False):
        super().__init__()
        self.path = path
        self.fix_lua_indexing = fix_lua_indexing
        self.data_file = None
        self.read_index(path)

    def read_index(self, path):
        with open(index_file_path(path), "rb") as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, (
                "Index file doesn't match expected format. "
                "Make sure that --dataset-impl is configured properly."
            )
            version = f.read(8)
            assert struct.unpack("<Q", version) == (1,)
            code, self.element_size = struct.unpack("<QQ", f.read(16))
            self.dtype = dtypes[code]
            self._len, self.s = struct.unpack("<QQ", f.read(16))
            self.dim_offsets = read_longs(f, self._len + 1)
            self.data_offsets = read_longs(f, self._len + 1)
            self.sizes = read_longs(f, self.s)

    def read_data(self, path):
        self.data_file = open(data_file_path(path), "rb", buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError("index out of range")

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        if not self.data_file:
            self.read_data(self.path)
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i] : self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        self.data_file.seek(self.data_offsets[i] * self.element_size)
        self.data_file.readinto(a)
        item = torch.from_numpy(a).long()
        if self.fix_lua_indexing:
            item -= 1  # subtract 1 for 0-based indexing
        return item

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(
            data_file_path(path)
        )

    @property
    def supports_prefetch(self):
        return False  # avoid prefetching to save memory


class MusicMassDataset(torch.utils.data.Dataset):
    """Masked Language Pair dataset (only support for single language)
       [x1, x2, x3, x4, x5]
                 |
                 V
       src: [x1, _, _, x4, x5]
       tgt: [x1, x2] => [x2, x3]
    """

    def __init__(
        self, src, sizes, vocab,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, lang_id=None, ratio=None, training=True,
        pred_probs=None, lang="",
    ):
        self.src = src
        self.sizes = np.array(sizes)
        self.vocab = vocab
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.lang_id = lang_id
        self.ratio = ratio
        self.training = training
        self.pred_probs = pred_probs

        self.sep_token = vocab.nspecial
        self.align_token = self.sep_token + 1
        self.lang = lang
        self.mask_len_expect_per_segment = 10
        self.pitch_start = self.align_token + 1
        self.duration_start = self.align_token + 129

    def __getitem__(self, index):
        if self.training is False:
            src_item = self.src[index]
            src_list = src_item.tolist()
            sep_positions = [i for i, x in enumerate(src_list) if x == self.sep_token]
            sep_positions.insert(0, -1)
            source = []
            source_sent_ids = []
            for i in range(len(sep_positions)-1):
                sent = src_list[sep_positions[i] + 1: sep_positions[i + 1]]
                sent = [ch for ch in sent if ch != self.align_token]
                source.extend(sent)
                source_sent_ids.extend([i] * len(sent))
            source.append(self.vocab.eos_index)
            source_sent_ids.append(-1)  # -1 non word for lyric
            source.insert(0, self.vocab.eos_index)
            source_sent_ids.insert(0, -1)

            output = source[1:]
            target_sent_ids = source_sent_ids[1:]
            target = source[:-1]
        else:
            src_item = self.src[index]
            src_list = src_item.tolist()

            sep_positions = [
                i for i, x in enumerate(src_list) if x == self.sep_token
            ]
            sep_positions.insert(0, -1)

            s = []
            source_sent_ids = []
            for i in range(len(sep_positions)-1):
                sent = src_list[sep_positions[i] + 1:sep_positions[i + 1]]
                sent = [ch for ch in sent if ch != self.align_token]
                s.extend(sent)
                source_sent_ids.extend([i] * len(sent))

            segment_num = round(len(s) / (
                self.mask_len_expect_per_segment / self.ratio
            ))
            segment_num = max(1, segment_num)
            seg_len = len(s) // segment_num

            source = []
            output = []
            target = []
            target_sent_ids = []

            for i in range(segment_num):
                seg_start = i * seg_len
                seg_end = (i+1) * seg_len
                if i == segment_num - 1:
                    seg_end = len(s)
                if self.lang == 'melody':
                    assert len(s) % 2 == 0
                    if seg_start % 2 == 1:
                        seg_start -= 1
                    if seg_end % 2 == 1:
                        seg_end -= 1
                    mask_start, mask_length = self.mask_interval_align(seg_start, seg_end)
                else:
                    mask_start, mask_length = self.mask_interval(seg_start, seg_end)

                output.extend(s[mask_start: mask_start + mask_length].copy())

                for j in range(mask_start, mask_start + mask_length):
                    target_sent_ids.append(source_sent_ids[j])
                if mask_start == 0:
                    t = [self.vocab.eos_index] + s[mask_start: mask_start + mask_length - 1].copy()
                else:
                    t = s[mask_start - 1: mask_start + mask_length - 1].copy()

                if self.lang == 'lyric':
                    for w in t:
                        target.append(self.random_word(w, self.pred_probs))
                    for i in range(seg_start, seg_end):
                        w = s[i]
                        if i >= mask_start and i < mask_start + mask_length:
                            w = self.mask_word(w)
                        if w is not None:
                            source.append(w)
                else:
                    t = t[1:] + [t[0]]
                    t2 = []
                    for i in range(0, len(t), 2):
                        pit, dur = self.random_pitch_duration(t[i], t[i + 1], self.pred_probs)
                        t2.append(pit)
                        t2.append(dur)
                    t = [t2[-1]] + t2[:-1]
                    target.extend(t)

                    assert seg_start % 2 == 0
                    assert seg_end % 2 == 0
                    for i in range(seg_start, seg_end, 2):
                        pit = s[i]
                        dur = s[i+1]
                        if i >= mask_start and i + 1 < mask_start + mask_length:
                            pit, dur = self.mask_pitch_duration(pit, dur) 
                        if pit is not None and dur is not None:
                            source.append(pit)
                            source.append(dur)

            source.append(self.vocab.eos_index)
            source_sent_ids.append(-1)
            assert len(output) == len(target)
            assert len(source_sent_ids) == len(source)
            assert len(target_sent_ids) == len(target)

        return {
            'id': index,
            'source': torch.LongTensor(source),
            'target': torch.LongTensor(target),
            'output': torch.LongTensor(output),
            'source_sent_ids': torch.LongTensor(source_sent_ids),
            'target_sent_ids': torch.LongTensor(target_sent_ids)
        }

    def __len__(self):
        return len(self.src)

    def _collate(self, samples, pad_idx, eos_idx, segment_label):

        def merge(key, left_pad):
            return collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad,
            )

        def merge_sentId(key, left_pad, pad_idx=pad_idx):
            return collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad,
            )

        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge('source', left_pad=self.left_pad_source)
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        ntokens = sum(len(s['target']) for s in samples)

        prev_output_tokens = merge('target', left_pad=self.left_pad_target)
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

        target = merge('output', left_pad=self.left_pad_target)
        target = target.index_select(0, sort_order)

        source_sent_ids = merge_sentId(
            'source_sent_ids', left_pad=self.left_pad_target, pad_idx=-1
        )
        source_sent_ids = source_sent_ids.index_select(0, sort_order)
        target_sent_ids = merge_sentId(
            'target_sent_ids', left_pad=self.left_pad_target, pad_idx=-2
        )
        target_sent_ids = target_sent_ids.index_select(0, sort_order)

        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths
            },
            'target': target,
        }
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
        batch['net_input']['source_sent_ids'] = source_sent_ids
        batch['net_input']['target_sent_ids'] = target_sent_ids
        return batch

    def collater(self, samples):
        return self._collate(
            samples,
            pad_idx=self.vocab.pad(),
            eos_idx=self.vocab.eos(),
            segment_label=self.lang_id,
        )

    def get_dummy_batch(
        self,
        num_tokens,
        max_positions,
        tgt_len=128
    ):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        source = self.vocab.dummy_sentence(tgt_len)
        target = self.vocab.dummy_sentence(tgt_len)
        bsz = max(num_tokens // tgt_len, 1)
        return self.collater([
            {
                'id': i,
                'source': source,
                'target': target,
                'output': target,
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(self.sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False) and getattr(self.src, 'supports_prefetch', False)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)

    def size(self, index):
        return (self.sizes[index], int(round(self.sizes[index] * self.ratio)))

    def mask_word(self, w):
        p = np.random.random()
        if p >= 0.2:
            return self.vocab.mask_index
        elif p >= 0.1:
            return np.random.randint(self.vocab.nspecial+1, len(self.vocab))
        else:
            return w

    def random_word(self, w, pred_probs):
        cands = [
            self.vocab.mask_index,
            np.random.randint(self.vocab.nspecial+1, len(self.vocab)),
            w
        ]
        prob = torch.multinomial(self.pred_probs, 1, replacement=True)
        return cands[prob]

    def mask_pitch(self, w):
        p = np.random.random()
        if p >= 0.2:
            return self.vocab.mask_index
        elif p >= 0.1:
            return np.random.randint(self.pitch_start, self.duration_start)
        else:
            return w

    def random_pitch(self, w, pred_probs):
        cands = [
            self.vocab.mask_index,
            np.random.randint(self.pitch_start, self.duration_start),
            w
        ]
        prob = torch.multinomial(self.pred_probs, 1, replacement=True)
        return cands[prob]

    def mask_duration(self, w):
        p = np.random.random()
        if p >= 0.2:
            return self.vocab.mask_index
        elif p >= 0.1:
            return np.random.randint(self.duration_start, len(self.vocab))
        else:
            return w

    def random_duration(self, w, pred_probs):
        cands = [
            self.vocab.mask_index,
            np.random.randint(self.duration_start, len(self.vocab)),
            w
        ]
        prob = torch.multinomial(self.pred_probs, 1, replacement=True)
        return cands[prob]

    def mask_pitch_duration(self, pit, dur):
        p = np.random.random()
        if p >= 0.2:
            return self.vocab.mask_index, self.vocab.mask_index
        elif p >= 0.1:
            ret_pit = np.random.randint(self.pitch_start, self.duration_start)
            ret_dur = np.random.randint(self.duration_start, len(self.vocab))
            return ret_pit, ret_dur
        else:
            return pit, dur

    def random_pitch_duration(self, pit, dur, pred_probs):
        rnd_pit = np.random.randint(self.pitch_start, self.duration_start)
        rnd_dur = np.random.randint(self.duration_start, len(self.vocab))
        cands = [
            (self.vocab.mask_index, self.vocab.mask_index),
            (rnd_pit, rnd_dur),
            (pit, dur)
        ]
        prob = torch.multinomial(self.pred_probs, 1, replacement=True)
        return cands[prob]

    def mask_interval(self, start, end):
        # not include end
        mask_length = round((end - start) * self.ratio)
        mask_length = max(1, mask_length)
        mask_start = self.mask_start(start, end - mask_length)
        return mask_start, mask_length

    def mask_start(self, start, end):
        p = np.random.random()
        if p >= 0.8:
            return start
        elif p >= 0.6:
            return end
        else:
            return np.random.randint(start, end + 1)

    def mask_interval_align(self, start, end):
        # not include end
        mask_length = round((end-start) * self.ratio)
        if mask_length % 2 != 0:
            mask_length -= 1
        mask_length = max(2, mask_length)
        mask_start = self.mask_start(start, end - mask_length)
        if mask_start % 2 != 0:
            mask_start -= 1
        return mask_start, mask_length


class MusicMtDataset(torch.utils.data.Dataset):
    """
    [x1, x2, x3, x4, x5] [y1, y2, y3, y4, y5]
                        |
                        V
    [x1, x2, x3, x4, x5] [y1, y2, y3, y4, y5]
    """
    def __init__(
        self, src, src_sizes, tgt, tgt_sizes, src_vocab, tgt_vocab,
        src_lang_id, tgt_lang_id,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        src_lang="",
        tgt_lang="",
    ):
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes)
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        assert self.src_vocab.nspecial == self.tgt_vocab.nspecial

        self.sep_token = self.src_vocab.nspecial
        self.align_token = self.sep_token + 1

    def __getitem__(self, index):
        tgt_item = self.tgt[index]
        src_item = self.src[index]

        src_list = src_item.tolist()
        tgt_list = tgt_item.tolist()

        # For Source
        sep_positions = [i for i, x in enumerate(src_list) if x == self.sep_token]
        sep_positions.insert(0, -1)
        sentences = []
        for i in range(len(sep_positions)-1):
            sentences.append(src_list[sep_positions[i]+1:sep_positions[i+1]])

        source = []
        source_sent_ids = []
        source_word_ids = []
        word_idx = 0
        for i, s in enumerate(sentences):
            for t in s:
                if t == self.align_token:
                    word_idx += 1
                else:
                    source.append(t)
                    source_word_ids.append(word_idx)
                    source_sent_ids.append(i)

            source.append(self.sep_token)
            source_sent_ids.append(i)
            source_word_ids.append(word_idx)
            word_idx += 1

        source.append(self.src_vocab.eos_index)
        source_sent_ids.append(-1)
        source_word_ids.append(word_idx)

        assert len(source) == len(source_sent_ids)
        assert len(source) == len(source_word_ids)

        # For Target
        sep_positions = [i for i, x in enumerate(tgt_list) if x == self.sep_token]
        sep_positions.insert(0, -1)
        sentences = []
        for i in range(len(sep_positions) - 1):
            sentences.append(
                tgt_list[sep_positions[i] + 1: sep_positions[i + 1]]
            )

        target = []
        target_sent_ids = []
        target_word_ids = []
        word_idx = 0
        for i, s in enumerate(sentences):
            for t in s:
                if t == self.align_token:
                    word_idx += 1
                else:
                    target.append(t)
                    target_word_ids.append(word_idx)
                    target_sent_ids.append(i)

            target.append(self.sep_token)  # Add sep token for every sentence
            target_sent_ids.append(i)
            target_word_ids.append(word_idx)
            word_idx += 1

        target.append(self.tgt_vocab.eos_index)
        target_sent_ids.append(-2)  # -2 for tgt non words
        target_word_ids.append(word_idx)  # eos token word align
        assert len(target) == len(target_sent_ids)
        assert len(target) == len(target_word_ids)

        return {
            'id': index,
            'source': torch.LongTensor(source),
            'target': torch.LongTensor(target),
            'source_sent_ids': torch.LongTensor(source_sent_ids),
            'target_sent_ids': torch.LongTensor(target_sent_ids),
            'source_word_ids': torch.LongTensor(source_word_ids),
            'target_word_ids': torch.LongTensor(target_word_ids)
        }

    def __len__(self):
        return len(self.src)

    def collate(
        self,
        samples,
        pad_idx,
        eos_idx,
        left_pad_source=True,
        left_pad_target=False,
        input_feeding=True
    ):
        if len(samples) == 0:
            return {}

        def merge(key, left_pad, move_eos_to_beginning=False):
            return collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad, move_eos_to_beginning,
            )

        def merge_sentId(key, left_pad, pad_idx=pad_idx):
            return collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad,
            )

        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge('source', left_pad=left_pad_source)
        # sort by descending source length
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        source_sent_ids = None
        source_sent_ids = merge_sentId(
            'source_sent_ids', left_pad=self.left_pad_target, pad_idx=-1
        )
        source_sent_ids = source_sent_ids.index_select(0, sort_order)
        source_word_ids = merge_sentId(
            'source_word_ids', left_pad=self.left_pad_target, pad_idx=-1
        )
        source_word_ids = source_word_ids.index_select(0, sort_order)

        prev_output_tokens = None
        target = None
        target_sent_ids = None

        if samples[0].get('target', None) is not None:
            target = merge('target', left_pad=left_pad_target)
            target = target.index_select(0, sort_order)
            ntokens = sum(len(s['target']) for s in samples)

            target_sent_ids = merge_sentId(
                'target_sent_ids', left_pad=self.left_pad_target, pad_idx=-2
            )
            target_sent_ids = target_sent_ids.index_select(0, sort_order)
            target_word_ids = merge_sentId(
                'target_word_ids', left_pad=self.left_pad_target, pad_idx=-2
            )
            target_word_ids = target_word_ids.index_select(0, sort_order)

            if input_feeding:
                # we create a shifted version of targets for feeding the
                # previous output token(s) into the next decoder step
                prev_output_tokens = merge(
                    'target',
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                )
                prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        else:
            ntokens = sum(len(s['source']) for s in samples)

        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
            'target': target,
            'word_ids': {
                'source_word_ids': source_word_ids,
                'target_word_ids': target_word_ids,
            }
        }
        if prev_output_tokens is not None:
            batch['net_input']['prev_output_tokens'] = prev_output_tokens
        if source_sent_ids is not None:
            batch['net_input']['source_sent_ids'] = source_sent_ids
        if target_sent_ids is not None:
            batch['net_input']['target_sent_ids'] = target_sent_ids
        return batch

    def generate_dummy_batch(self, num_tokens, collate_fn, src_vocab, tgt_vocab, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        bsz = num_tokens // max(src_len, tgt_len)
        return collate_fn([
            {
                'id': i,
                'source': src_vocab.dummy_sentence(src_len),
                'target': tgt_vocab.dummy_sentence(tgt_len),
                'output': tgt_vocab.dummy_sentence(tgt_len),
            }
            for i in range(bsz)
        ])

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return self.collate(
            samples, pad_idx=self.src_vocab.pad(), eos_idx=self.src_vocab.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        return self.generate_dummy_batch(num_tokens, self.collater, self.src_vocab, self.tgt_vocab, src_len, tgt_len)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False) and
            (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


def _match_types(arg1, arg2):
    """Convert the numerical argument to the same type as the other argument"""

    def upgrade(arg_number, arg_structure):
        if isinstance(arg_structure, tuple):
            return tuple([arg_number] * len(arg_structure))
        elif isinstance(arg_structure, dict):
            arg = copy.deepcopy(arg_structure)
            for k in arg:
                arg[k] = upgrade(arg_number, arg_structure[k])
            return arg
        else:
            return arg_number

    if isinstance(arg1, float) or isinstance(arg1, int):
        return upgrade(arg1, arg2), arg2
    elif isinstance(arg2, float) or isinstance(arg2, int):
        return arg1, upgrade(arg2, arg1)

    return arg1, arg2


def resolve_max_positions(*args):
    """Resolve max position constraints from multiple sources."""

    def map_value_update(d1, d2):
        updated_value = copy.deepcopy(d1)
        for key in d2:
            if key not in updated_value:
                updated_value[key] = d2[key]
            else:
                updated_value[key] = min(d1[key], d2[key])
        return updated_value

    def nullsafe_min(l):
        minim = None
        for item in l:
            if minim is None:
                minim = item
            elif item is not None and item < minim:
                minim = item
        return minim

    max_positions = None
    for arg in args:
        if max_positions is None:
            max_positions = arg
        elif arg is not None:
            max_positions, arg = _match_types(max_positions, arg)
            if isinstance(arg, float) or isinstance(arg, int):
                max_positions = min(max_positions, arg)
            elif isinstance(arg, dict):
                max_positions = map_value_update(max_positions, arg)
            else:
                max_positions = tuple(map(nullsafe_min, zip(max_positions, arg)))

    return max_positions
