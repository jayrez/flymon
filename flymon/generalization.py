"""Instance/seed-disjoint nested nearest-centroid analysis, with auditable folds."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist
from .dataset import SIZES


def instance_blocks(labels):
    groups=[np.flatnonzero(labels==c) for c in np.unique(labels)]
    return [np.array([g[j] for g in groups if len(g)>j],int) for j in range(max(map(len,groups)))]


def split_plan(labels, nseed=20, blocks=None):
    blocks=instance_blocks(labels) if blocks is None else blocks
    seedblocks=np.array_split(np.arange(nseed),4)
    folds=[]
    for bi,test_i in enumerate(blocks):
        train_i=np.setdiff1d(np.arange(len(labels)),test_i)
        remaining=[(j,b) for j,b in enumerate(blocks) if j!=bi]
        for si,test_s in enumerate(seedblocks):
            train_s=np.setdiff1d(np.arange(nseed),test_s)
            available=[s for j,s in enumerate(seedblocks) if j!=si]
            inner=[]
            for k,(bj,valid_i) in enumerate(remaining):
                valid_s=available[k%len(available)]
                inner.append(dict(train_i=np.setdiff1d(train_i,valid_i).tolist(),
                                  train_s=np.setdiff1d(train_s,valid_s).tolist(),
                                  test_i=valid_i.tolist(),test_s=valid_s.tolist()))
            folds.append(dict(instance_block=bi,seed_block=si,train_i=train_i.tolist(),
                              test_i=test_i.tolist(),train_s=train_s.tolist(),test_s=test_s.tolist(),inner=inner))
    return folds


@dataclass
class PreparedFold:
    train_i: np.ndarray
    test_i: np.ndarray
    test_s: np.ndarray
    sums: np.ndarray
    square_total: np.ndarray
    repeats: int
    test: np.ndarray

    @classmethod
    def make(cls,x,f):
        ti=np.array(f['train_i']);vi=np.array(f['test_i']);ts=np.array(f['train_s']);vs=np.array(f['test_s'])
        train=x[ti][:,ts].astype(np.float64)
        return cls(ti,vi,vs,train.sum(axis=1),np.square(train).sum(axis=(0,1)),len(ts),
                   x[vi][:,vs].reshape(-1,x.shape[-1]).astype(np.float64))

    def fit(self,y,nclass):
        """Only training instance labels and sufficient statistics are accessible."""
        means=[];counts=[]
        for c in range(nclass):
            which=y[self.train_i]==c
            n=int(which.sum())*self.repeats
            if not n:raise ValueError('Training class missing')
            means.append(self.sums[which].sum(axis=0)/n);counts.append(n)
        means=np.stack(means);counts=np.array(counts)
        grand=np.average(means,axis=0,weights=counts)
        between=(counts[:,None]*np.square(means-grand)).sum(axis=0)/(nclass-1)
        within=np.maximum(0,self.square_total-(counts[:,None]*np.square(means)).sum(axis=0))/(counts.sum()-nclass)
        ranking=np.argsort(-between/(within+1e-6),kind='stable')
        return means,ranking

    def predictions(self,means,ranking,sizes):
        # Cumulative feature contributions evaluate every population size in one pass.
        delta=np.square(self.test[:,None,ranking]-means[None,:,ranking])
        cumulative=np.cumsum(delta,axis=-1)
        return np.stack([np.argmin(cumulative[:,:,min(s,len(ranking))-1],axis=1) for s in sizes])

    def truth(self,y):
        return np.repeat(y[self.test_i],len(self.test_s))


def prepare_plan(x,plan):
    return [(PreparedFold.make(x,f),[PreparedFold.make(x,i) for i in f['inner']]) for f in plan]


def nested_fixed_subsets(y,plan,prepared,subsets,order,shape,nclass=None):
    """Nested selection of one *label-free* feature subset per outer fold.

    subsets: {name: feature-index array}, each a fixed anatomically chosen subset.
    order: subset names for deterministic tie-breaking (earlier name preferred).
    For every outer fold the inner folds pick the subset with the highest pooled
    inner-training accuracy; the chosen subset is then evaluated once on the outer
    held-out instances/seeds. Test labels never influence the choice. `prepared`
    is the prepare_plan(x, plan) output (label-independent), so this is cheap to
    repeat under permuted y for a valid permutation null."""
    nclass=len(np.unique(y)) if nclass is None else nclass
    rank={n:i for i,n in enumerate(order)}
    predictions=np.full(shape,-1,int);choices=[]
    for f,(outer,inners) in zip(plan,prepared):
        correct={n:0 for n in subsets};total=0
        for inner in inners:
            means,_=inner.fit(y,nclass);truth=inner.truth(y)
            for n,idx in subsets.items():
                pred=np.argmin(cdist(inner.test[:,idx],means[:,idx],'sqeuclidean'),axis=1)
                correct[n]+=int((pred==truth).sum())
            total+=len(truth)
        chosen=min(subsets,key=lambda n:(-correct[n],rank[n]))
        means,_=outer.fit(y,nclass);idx=subsets[chosen]
        pred=np.argmin(cdist(outer.test[:,idx],means[:,idx],'sqeuclidean'),axis=1)
        predictions[np.ix_(outer.test_i,outer.test_s)]=pred.reshape(len(outer.test_i),len(outer.test_s))
        choices.append(dict(instance_block=f['instance_block'],seed_block=f['seed_block'],chosen=chosen,
                            chosen_size=int(len(idx)),inner_accuracy={n:correct[n]/total for n in subsets}))
    assert np.all(predictions>=0)
    return predictions,choices


def nested(x,y,plan,sizes=SIZES,prepared=None,details=True,frozen=None):
    nclass=len(np.unique(y));prepared=prepare_plan(x,plan) if prepared is None else prepared
    predictions=np.full(x.shape[:2],-1,int)
    fixed=np.full((len(sizes),*x.shape[:2]),-1,int) if details else None
    frozen_pred=np.full(x.shape[:2],-1,int) if frozen is not None else None
    choices=[];fit_count=0
    for f,(outer,inners) in zip(plan,prepared):
        correct=np.zeros(len(sizes),int);total=0
        for inner in inners:
            means,ranking=inner.fit(y,nclass);fit_count+=1
            p=inner.predictions(means,ranking,sizes)
            correct+=(p==inner.truth(y)[None]).sum(axis=1);total+=p.shape[1]
        chosen=int(np.argmax(correct))
        means,ranking=outer.fit(y,nclass);fit_count+=1
        p=outer.predictions(means,ranking,sizes)
        idx=np.ix_(outer.test_i,outer.test_s)
        predictions[idx]=p[chosen].reshape(len(outer.test_i),len(outer.test_s))
        if details:
            for k in range(len(sizes)):fixed[k][idx]=p[k].reshape(len(outer.test_i),len(outer.test_s))
            choices.append(dict(instance_block=f['instance_block'],seed_block=f['seed_block'],
                population=int(sizes[chosen]),inner_accuracy={str(s):float(c/total) for s,c in zip(sizes,correct)},
                selected=ranking[:min(sizes[chosen],x.shape[-1])].tolist(),top50=ranking[:50].tolist()))
        if frozen is not None:
            fp=np.argmin(cdist(outer.test[:,frozen],means[:,frozen],'sqeuclidean'),axis=1)
            frozen_pred[idx]=fp.reshape(len(outer.test_i),len(outer.test_s))
    assert np.all(predictions>=0)
    return dict(predictions=predictions,fixed=fixed,frozen=frozen_pred,choices=choices,
                accuracy=float((predictions==y[:,None]).mean()),ranking_fits=fit_count)


def unselected(x,y,plan):
    p=np.full(x.shape[:2],-1,int)
    for f in plan:
        training=x[f['train_i']][:,f['train_s']]
        means=np.stack([training[y[f['train_i']]==c].mean(axis=(0,1)) for c in range(len(np.unique(y)))])
        test=x[f['test_i']][:,f['test_s']]
        pred=np.argmin(cdist(test.reshape(-1,x.shape[-1]),means,'sqeuclidean'),axis=1)
        p[np.ix_(f['test_i'],f['test_s'])]=pred.reshape(len(f['test_i']),len(f['test_s']))
    assert np.all(p>=0)
    return p


def record(predictions,y,names):
    truth=np.broadcast_to(y[:,None],predictions.shape)
    conf=np.zeros((len(names),len(names)),int)
    np.add.at(conf,(truth.ravel(),predictions.ravel()),1)
    recalls=np.diag(conf)/conf.sum(axis=1)
    return dict(accuracy=float(np.trace(conf)/conf.sum()),balanced_accuracy=float(recalls.mean()),
                confusion=conf.tolist(),per_class_recall=dict(zip(names,recalls.tolist())),labels=list(names),
                chance=1/len(names),predictions=predictions.tolist())


def audit_plan(plan,records,seeds):
    checks=dict(image_overlap=True,instance_overlap=True,seed_overlap=True,inner_containment=True)
    def check(f):
        a={records[i]['instance_id'] for i in f['train_i']};b={records[i]['instance_id'] for i in f['test_i']}
        ha={h for i in f['train_i'] for h in records[i]['frame_sha256']}
        hb={h for i in f['test_i'] for h in records[i]['frame_sha256']}
        checks['instance_overlap'] &= not bool(a&b)
        checks['image_overlap'] &= not bool(ha&hb)
        checks['seed_overlap'] &= not bool(set(f['train_s'])&set(f['test_s']))
    for f in plan:
        check(f)
        for inner in f['inner']:
            check(inner)
            checks['inner_containment'] &= (set(inner['train_i']+inner['test_i'])<=set(f['train_i']) and
                                            set(inner['train_s']+inner['test_s'])<=set(f['train_s']))
    assert all(checks.values()),checks
    return checks


def bootstrap(predictions,y,count=10000):
    rng=np.random.default_rng(6062027);scores=(predictions==y[:,None]).mean(axis=1)
    groups=[np.flatnonzero(y==c) for c in np.unique(y)]
    draws=np.concatenate([rng.choice(g,(count,len(g)),replace=True) for g in groups],axis=1)
    null=scores[draws].mean(axis=1)
    return dict(unit='whole visual instance, stratified within class; all out-of-fold seed predictions retained',
                resamples=count,interval95=np.quantile(null,[.025,.975]).tolist(),
                number_of_instance_units=len(scores),first_draw_instance_indices=draws[0].tolist())


def permutation_labels(y,blocks,rng):
    shuffled=y.copy()
    for block in blocks:shuffled[block]=rng.permutation(y[block])
    return shuffled
