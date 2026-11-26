import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_official_test as w


def test_commands(tmp_path):
    plan, analysis = w.commands(*[tmp_path / s for s in ('ckpt','data','cal','manifest','out')])
    assert len(plan) == 10
    for label, cmd in zip(w.CONFIGURATIONS, plan):
        assert cmd[cmd.index('--split')+1] == 'test'
        assert cmd[cmd.index('--batch-size')+1] == '16'
        assert '--max-images' not in cmd and '--benchmark' not in cmd
        flag = '--variant' if label in ('dense','unmasked') else '--label'
        assert cmd[cmd.index(flag)+1] == label
        assert cmd[cmd.index('--output')+1].endswith(label+'.json')
        assert cmd[cmd.index('--predictions-output')+1].endswith(label+'.csv')
    assert analysis[analysis.index('--configurations')+1:analysis.index('--repetitions')] == list(w.CONFIGURATIONS)


def test_approval_gate():
    w.check_approval(False, None, None, 'abc', 'dirty')
    for a,r,d in [(None,'abc',''), ('yes','wrong',''), ('yes','abc','dirty')]:
        with pytest.raises(ValueError):
            w.check_approval(True,a,r,'abc',d)
    w.check_approval(True,'team approval','abc','abc','')


def test_canonical_json(tmp_path):
    a,b = tmp_path/'a',tmp_path/'b'
    a.write_bytes(b'{"b": 2,\r\n "a": 1}')
    b.write_bytes(b'{"a":1,"b":2}')
    assert w.canonical_hash(a)==w.canonical_hash(b)
    assert w.raw_hash(a)!=w.raw_hash(b)


def test_hash_gate(tmp_path, monkeypatch):
    a,b,c = [tmp_path/x for x in ('a','b','c')]
    a.write_bytes(b'model'); b.write_text('{}'); c.write_text('{"x":1}')
    monkeypatch.setattr(w,'CHECKPOINT_HASH',w.raw_hash(a))
    monkeypatch.setattr(w,'CALIBRATION_HASH',w.canonical_hash(b))
    monkeypatch.setattr(w,'MANIFEST_HASH',w.canonical_hash(c))
    w.check_inputs(a,b,c)
    c.write_text('{}')
    with pytest.raises(ValueError): w.check_inputs(a,b,c)


def test_result_gate():
    d={'split':'test','scores':{'n':3669},'variant':'dense','layer_indices':None,'mask_config':None}
    w.check_result(d,'dense',{})
    with pytest.raises(ValueError): w.check_result({**d,'scores':{'n':32}},'dense',{})
    c={'p':.87,'top_k':4,'similarity_threshold':None}
    p={**d,'variant':'A','layer_indices':[3,6,9],'mask_config':{**c,'method':'top_p','grid_size':14,'tile_size':2}}
    w.check_result(p,'A_50',{'A_50':c})
    with pytest.raises(ValueError): w.check_result({**p,'layer_indices':[9]},'A_50',{'A_50':c})
