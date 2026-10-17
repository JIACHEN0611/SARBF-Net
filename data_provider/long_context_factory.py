from copy import copy
from data_provider.data_factory import data_provider
from data_provider.split_geometry import global_target_start, source_row_count


MODEL_SEQ_LEN = 96
LONG_CONTEXT_LEN = 768


def make_data_args(args):
    if int(args.seq_len) != MODEL_SEQ_LEN:
        raise ValueError("LCM v1 requires model seq_len=96")
    if int(args.long_len) != LONG_CONTEXT_LEN:
        raise ValueError("LCM v1 requires long_len=768")
    data_args = copy(args)
    data_args.seq_len = LONG_CONTEXT_LEN
    return data_args


def long_context_data_provider(args, flag, test_batch_size=None):
    data_args = make_data_args(args)
    dataset, loader = data_provider(
        data_args, flag, test_batch_size=test_batch_size
    )
    dataset._lcm_target_start = _global_target_start(data_args, flag)
    return dataset, loader


def _global_target_start(data_args, flag):
    from pathlib import Path

    path = Path(data_args.root_path) / data_args.data_path
    row_count = source_row_count(path)
    return global_target_start(data_args.data, flag, row_count, LONG_CONTEXT_LEN)


def target_indices_for_dataset(dataset):
    if not hasattr(dataset, "_lcm_target_start"):
        raise ValueError("dataset was not created by long_context_data_provider")
    start = int(dataset._lcm_target_start)
    return list(range(start, start + len(dataset)))
