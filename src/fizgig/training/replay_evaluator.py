"""6C.2 CPU metric reference. No model, selector or trainer dependencies."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from collections import defaultdict
import numpy as np

DOMAINS = ('target', 'near', 'far')

@dataclass(frozen=True, order=True)
class Case:
    domain: str
    unit: str      # independent source image/concept family; aliases stay in same unit
    variant: str   # prompt variant within unit
    timestep: float
    noise: int
    input_hash: str  # digest of exact latent, noise, t, conditioning and model identity

    def __post_init__(self):
        if self.domain not in DOMAINS or not self.unit or not self.variant or not 0 < self.timestep < 1:
            raise ValueError('invalid case')
        if len(self.input_hash) != 64 or any(c not in '0123456789abcdef' for c in self.input_hash):
            raise ValueError('input_hash must be lowercase SHA256')


def freeze(cases):
    cases = tuple(sorted(cases))
    if not cases or len(set(cases)) != len(cases):
        raise ValueError('empty or duplicate manifest')
    ids = [(c.domain,c.unit,c.variant,c.timestep,c.noise) for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate logical case')
    if set(c.domain for c in cases) != set(DOMAINS):
        raise ValueError('all three domains required')
    # A source unit may not span evaluation domains; use separately curated panels.
    if len({(c.domain,c.unit) for c in cases}) != len({c.unit for c in cases}):
        raise ValueError('unit crosses domains')
    # Prevent silent unequal timestep/noise coverage across units and variants.
    design = defaultdict(set)
    for c in cases:
        design[c.domain,c.unit,c.variant].add((c.timestep,c.noise))
    coverage = list(design.values())
    if any(x != coverage[0] for x in coverage):
        raise ValueError('unbalanced time/noise coverage')
    digest = hashlib.sha256(json.dumps([asdict(c) for c in cases],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return cases,digest


def _array(x):
    a = np.asarray(x)
    if a.dtype.kind not in 'fiu' or not a.size or not np.isfinite(a).all():
        raise ValueError('nonfinite, empty or nonnumeric prediction')
    return a.astype(np.float64,copy=False)


def _mse(x):
    return float(np.mean(x*x,dtype=np.float64))


def metrics(case, candidate, parent, base, target=None):
    """Per-case MSE in native prediction coordinates, averaged over elements.

    Signed improvement/deterioration uses residual expansion, not subtracting
    rounded scalar losses. References are exact observed prediction tensors.
    """
    p,r,b = map(_array,(candidate,parent,base))
    if p.shape != r.shape or r.shape != b.shape:
        raise ValueError('prediction shape mismatch (no broadcasting)')
    d = p-r
    rb = r-b
    out = {'parent_mse':_mse(d), 'base_mse':_mse(p-b),
           'parent_base_mse':_mse(rb), 'base_increment':float(np.mean(d*(2*rb+d))),
           'base_energy':_mse(b)}
    if case.domain == 'target':
        if target is None: raise ValueError('missing exact flow target')
        y = _array(target)
        if y.shape != p.shape: raise ValueError('target shape mismatch')
        ry = r-y
        out.update(target_mse=_mse(p-y), parent_target_mse=_mse(ry),
                   target_gain=-float(np.mean(d*(2*ry+d))))
    elif target is not None:
        raise ValueError('target supplied to preservation-only domain')
    if not all(math.isfinite(v) for v in out.values()): raise ValueError('metric overflow')
    return out


class Evaluator:
    """One candidate at a time; only scalar rows retained here.

    Caller supplies immutable reference tensors/cache and checked input hashes.
    A failed/missing case invalidates the candidate; never average subsets.
    """
    def __init__(self,cases):
        self.cases,self.manifest_hash=freeze(cases)
        self.expected=set(self.cases)
        self.rows={}
        self.reference_hashes={}

    def add(self,case,*,candidate,parent,base,target=None):
        if case not in self.expected or case in self.rows:
            raise ValueError('unknown or duplicate case')
        self.rows[case]=metrics(case,candidate,parent,base,target)
        h=hashlib.sha256()
        for ref in (parent,base,target):
            if ref is None:
                h.update(b'none')
                continue
            a=np.ascontiguousarray(_array(ref),dtype='<f8')
            h.update(json.dumps(list(a.shape)).encode()+b':')
            h.update(memoryview(a).cast('B'))
        self.reference_hashes[case]=h.hexdigest()

    def finish(self):
        if set(self.rows)!=self.expected: raise ValueError('incomplete candidate')
        return {'manifest_hash':self.manifest_hash,'domains':aggregate(self.rows),
                'rows':[{'case':asdict(c),'reference_hash':self.reference_hashes[c],**self.rows[c]} for c in self.cases]}


def aggregate(rows):
    """Equal domain units, equal variants within unit, equal time/noise within variant.

    Also keep per-timestep and per-unit results; no mixing target/near/far.
    """
    def combine(selected):
        grouped=defaultdict(lambda:defaultdict(list))
        for c,m in selected: grouped[c.unit][c.variant].append(m)
        keys=next(iter(next(iter(grouped.values())).values()))[0].keys()
        unit={u:{k:float(np.mean([np.mean([r[k] for r in v]) for v in vs.values()])) for k in keys} for u,vs in grouped.items()}
        mean={k:float(np.mean([v[k] for v in unit.values()])) for k in keys}
        return {'mean':mean,'units':unit,'worst_unit_base_increment':max(v['base_increment'] for v in unit.values())}
    out={}
    for domain in DOMAINS:
        pairs=[(c,m) for c,m in rows.items() if c.domain==domain]
        out[domain]=combine(pairs)
        out[domain]['timesteps']={str(t):combine([(c,m) for c,m in pairs if c.timestep==t]) for t in sorted({c.timestep for c,m in pairs})}
    return out


def objectives(result):
    d=result['domains']
    # All three oriented as higher is better; no combining units or domains.
    return {'target_gain':d['target']['mean']['target_gain'],
            'near_preserve':-d['near']['mean']['base_increment'],
            'far_preserve':-d['far']['mean']['base_increment']}


def pareto(a,b):
    if set(a)!=set(b) or not a or not all(math.isfinite(v) for v in [*a.values(),*b.values()]):
        raise ValueError('invalid objectives')
    return all(a[k]>=b[k] for k in a) and any(a[k]>b[k] for k in a)


def paired_summary(differences, *, seed=6102, draws=20000):
    """One number per independent TRAINING seed (not image, t, or checkpoint).

    Descriptive at n<12. Percentile bootstrap is exploratory at n>=12, not a
    simultaneous confirmatory interval. Exact sign-test p ignores magnitudes.
    """
    x=_array(differences)
    if x.ndim!=1: raise ValueError('one scalar per independent seed')
    nonzero=x[x!=0]; n=len(nonzero); pos=int((nonzero>0).sum())
    p=min(1.,2*sum(math.comb(n,i) for i in range(min(pos,n-pos)+1))/2**n) if n else 1.
    out={'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),
         'positive':pos,'negative':n-pos,'ties':len(x)-n,'sign_p_two_sided':p,'exploratory_ci95':None}
    if len(x)>=12:
        rng=np.random.default_rng(seed)
        means=x[rng.integers(0,len(x),(draws,len(x)))].mean(axis=1)
        out['exploratory_ci95']=[float(v) for v in np.quantile(means,[.025,.975])]
    return out


def gate(lower_bounds, margins):
    """Caller supplies pre-specified simultaneous paired one-sided bounds.
    All differences are oriented Adaptive minus control, higher is better.
    This is noninferiority on preservation, NOT equivalence.
    """
    keys={'target_gain','near_preserve','far_preserve'}
    if set(lower_bounds)!=keys or set(margins)!=keys or any(not math.isfinite(v) for v in [*lower_bounds.values(),*margins.values()]) or any(v<0 for v in margins.values()):
        raise ValueError('invalid bounds/margins')
    return (lower_bounds['target_gain']>margins['target_gain'] and
            lower_bounds['near_preserve']>=-margins['near_preserve'] and
            lower_bounds['far_preserve']>=-margins['far_preserve'])


def compare_candidates(candidate, control):
    """Same parent and frozen reference inputs required; no independent-noise comparisons."""
    if candidate['manifest_hash']!=control['manifest_hash']:
        raise ValueError('different evaluation manifests')
    if len(candidate['rows'])!=len(control['rows']): raise ValueError('different row counts')
    for a,b in zip(candidate['rows'],control['rows']):
        if a['case']!=b['case'] or a['reference_hash']!=b['reference_hash']:
            raise ValueError('different inputs or reference predictions')
    a,b=objectives(candidate),objectives(control)
    return {k:a[k]-b[k] for k in a}
