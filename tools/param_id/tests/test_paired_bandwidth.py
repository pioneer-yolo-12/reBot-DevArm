import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from friction_id.paired_bandwidth import equation,summaries


def fixture():
    q=np.arange(.01,.54,.002)
    d=dict(q=q,group=np.array(['trial']*len(q)),df=np.ones(len(q)),dv=.1+.02*np.sin(30*q),da=.2*np.cos(50*q))
    X=np.column_stack([d['df'],d['dv'],d['da'],np.sin(2*np.pi*q/.05),np.cos(2*np.pi*q/.05)])
    d['D']=X@np.array([.3,.5,.04,.1,-.1])
    return d


def test_filter_preserves_the_linear_equation_and_common_support():
    d=fixture();outputs=[equation(d,sigma=x,period=.05) for x in (0,.002,.004,.006)]
    for e in outputs:
        assert np.array_equal(e['q'],outputs[0]['q'])
        assert np.allclose(e['y'],e['X']@np.array([.3,.5,.04,.1,-.1]),atol=1e-12)
    # Filtering only y would not preserve this relation to the raw columns.
    assert np.max(abs(outputs[2]['y']-outputs[0]['X']@np.array([.3,.5,.04,.1,-.1])))>.001


def test_gaps_are_split_and_short_fragments_rejected():
    d=fixture();keep=(d['q']<.20)|(d['q']>.30)
    e=equation({k:v[keep] for k,v in d.items()})
    assert not np.any((e['q']>.181)&(e['q']<.317))
    with pytest.raises(ValueError):equation({k:v[:10] for k,v in d.items()})
    assert summaries(e)['roundtrip_std'] is None
