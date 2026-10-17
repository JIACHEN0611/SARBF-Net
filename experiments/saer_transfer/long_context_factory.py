"""Create aligned long-context loaders while preserving backbone input length."""

from copy import copy

from experiments.saer_transfer.data_factory import data_provider


SUPPORTED_MODEL_SEQ_LENS = {96, 336}
HISTORY_LEN = 672


def make_data_args(args):
    seq_len = int(args.seq_len)
    long_len = int(args.long_len)
    if seq_len not in SUPPORTED_MODEL_SEQ_LENS:
        raise ValueError("SA-LCM frontend requires seq_len in {96,336}")
    if long_len != seq_len + HISTORY_LEN:
        raise ValueError("SA-LCM frontend requires long_len=seq_len+672")
    data_args = copy(args)
    data_args.seq_len = long_len
    return data_args


def long_context_data_provider(args, flag):
    return data_provider(make_data_args(args), flag)
