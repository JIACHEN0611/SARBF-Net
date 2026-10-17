import json
from pathlib import Path
import math

import pytest

ROOT=Path(__file__).resolve().parents[1]


def summary_module():
    assert (ROOT/'tools/saer_transfer/summarize.py').is_file(), 'Transfer summary missing'
    from tools.saer_transfer import summarize
    return summarize


def write_run(root,seed,arm,mse,name=None):
    p=root/(name or f'{seed}-{arm}');p.mkdir()
    (p/'task.json').write_text(json.dumps(dict(backbone='DLinear',dataset='ETTm1',seq_len=96,
                        long_len=768,horizon=96,seed=seed,arm=arm)))
    (p/'metrics.json').write_text(json.dumps(dict(mse=mse,mae=mse/2,params=100,best_epoch=1)))
    return p


def test_summary_pairs_seeds_and_uses_sample_std(tmp_path):
    s=summary_module()
    for seed,original,saer in [(2021,.2,.1),(2022,.4,.3)]:
        write_run(tmp_path,seed,'original',original);write_run(tmp_path,seed,'sa_frontend',saer)
    rows=s.collect_rows(tmp_path);pairs=s.pair_rows(rows);out=s.aggregate_rows(pairs)[0]
    assert len(rows)==4 and len(pairs)==2
    assert out['OriginalMSEMean']==pytest.approx(.3)
    assert out['SAERMSEMean']==pytest.approx(.2)
    assert out['OriginalMSEStd']==pytest.approx(math.sqrt(.02))
    assert out['SAERMSEStd']==pytest.approx(math.sqrt(.02))
    assert out['MSEImprovementPct']==pytest.approx(33.333333)
    assert s.aggregate_rows(pairs[:1])[0]['OriginalMSEStd'] is None
    write_run(tmp_path,2021,'original',.2,'duplicate')
    with pytest.raises(ValueError,match='duplicate'):s.collect_rows(tmp_path)


def test_failed_and_unpaired_results_do_not_enter_pair_mean(tmp_path):
    s=summary_module()
    write_run(tmp_path,2021,'original',.2)
    failed=write_run(tmp_path,2021,'sa_frontend',.1)
    (failed/'failed.json').write_text('{}')
    assert len(s.collect_rows(tmp_path))==1
    assert s.pair_rows(s.collect_rows(tmp_path))==[]


def test_zero_baseline_keeps_raw_metrics_and_writes_blank_percentages(tmp_path):
    import csv
    s=summary_module()
    for seed,saer in [(2021,0.0),(2022,0.1)]:
        write_run(tmp_path,seed,'original',0.0)
        write_run(tmp_path,seed,'sa_frontend',saer)
    detail,paired,aggregate=[tmp_path/name for name in ['detail.csv','paired.csv','aggregate.csv']]
    assert s.write_summaries(tmp_path,detail,paired,aggregate)==(4,2,1)
    rows=list(csv.DictReader(paired.open()))
    assert all(r['MSEImprovementPct']==r['MAEImprovementPct']=='' for r in rows)
    assert all(r['BothImproved']=='False' for r in rows)
    row=list(csv.DictReader(aggregate.open()))[0]
    assert float(row['OriginalMSEMean'])==0
    assert float(row['SAERMSEMean'])==pytest.approx(.05)
    assert row['MSEImprovementPct']==row['MAEImprovementPct']==''
    assert row['MSEWins']==row['MAEWins']==row['BothImprovedSeeds']=='0'
