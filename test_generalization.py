"""Leakage, encoder-replay and inference-unit regression tests (CPU only)."""
import json
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.distance import cdist
from flymon.dataset import ROOT, RESULTS, digest, regression, load_dataset
from flymon.spatiotemporal import SpatiotemporalConfig, SpatialProjection
from flymon.generalization import (PreparedFold, split_plan, nested, audit_plan,
                                   bootstrap, permutation_labels, instance_blocks)


class GeneralizationTests(unittest.TestCase):
    def setUp(self):
        self.y=np.repeat(np.arange(5),5)
        self.x=np.random.default_rng(24).normal(size=(25,20,60))
        self.x[:,:,:5]+=np.eye(5)[self.y,None]*3
        self.plan=split_plan(self.y)

    def test_original_five_encoders_bit_exact(self):
        old=json.loads((ROOT/'results/experiment-03-spatiotemporal/trials.json').read_text())
        projection=SpatialProjection.__new__(SpatialProjection)
        projection.config=SpatiotemporalConfig(**old['config'])
        projection.ids=np.array(old['projection_neuron_indices'],np.int32)
        projection.grid_indices={};projection.populations={};offset=0
        for kind in ('LPLC2','LC10a'):
            for side in ('L','R'):
                xy=np.array(old['projection_grid_coordinates'][f'{kind}_{side}'])
                projection.grid_indices[kind,side]=tuple(xy.T)
                projection.populations[kind,side]=projection.ids[offset:offset+len(xy)];offset+=len(xy)
        self.assertEqual(len(regression(projection)),5)

    def test_dataset_immutable_and_unique(self):
        records,sequences=load_dataset()
        self.assertEqual(len(records),len(sequences))
        seen=set()
        for r in records:
            hashes=set(r['frame_sha256'])
            self.assertFalse(seen&hashes);seen.update(hashes)

    def test_all_outer_predictions_once_and_disjoint(self):
        records=[dict(instance_id=str(i),frame_sha256=[str(i)]) for i in range(len(self.y))]
        self.assertTrue(all(audit_plan(self.plan,records,range(20)).values()))
        seen=np.zeros(self.x.shape[:2],int)
        for f in self.plan:seen[np.ix_(f['test_i'],f['test_s'])]+=1
        np.testing.assert_array_equal(seen,np.ones_like(seen))

    def test_negative_leakage_is_rejected(self):
        records=[dict(instance_id=str(i),frame_sha256=[str(i)]) for i in range(len(self.y))]
        self.plan[0]['train_i']+=self.plan[0]['test_i'][:1]
        with self.assertRaises(AssertionError):audit_plan(self.plan,records,range(20))

    def test_score_matches_original_anova_on_balanced_data(self):
        f=self.plan[0];prepared=PreparedFold.make(self.x,f);means,rank=prepared.fit(self.y,5)
        training=self.x[f['train_i']][:,f['train_s']]
        grouped=np.stack([training[self.y[f['train_i']]==c].reshape(-1,60) for c in range(5)],axis=1)
        mean=grouped.mean(axis=0);grand=mean.mean(axis=0)
        between=grouped.shape[0]*np.square(mean-grand).sum(axis=0)/4
        within=np.square(grouped-mean[None]).sum(axis=(0,1))/(5*(grouped.shape[0]-1))
        expected=np.argsort(-between/(within+1e-6),kind='stable')
        np.testing.assert_array_equal(rank,expected)
        predicted=prepared.predictions(means,rank,(60,25,5))
        for i,size in enumerate((60,25,5)):
            direct=np.argmin(cdist(prepared.test[:,rank[:size]],means[:,rank[:size]],'sqeuclidean'),axis=1)
            np.testing.assert_array_equal(predicted[i],direct)

    def test_heldout_activity_and_labels_cannot_change_ranking_or_size(self):
        # Directly examine all inner fit/validation inputs, avoiding a false global-CV claim:
        # a test instance in one fold legitimately becomes training in another fold.
        f=self.plan[0];changed=self.x.copy();changed[f['test_i']]=1e8;changed[:,f['test_s']]=-1e8
        yy=self.y.copy();yy[f['test_i']]=(yy[f['test_i']]+1)%5
        for spec in [f,*f['inner']]:
            a=PreparedFold.make(self.x,spec);b=PreparedFold.make(changed,spec)
            ma,ra=a.fit(self.y,5);mb,rb=b.fit(yy,5)
            np.testing.assert_array_equal(ra,rb);np.testing.assert_array_equal(ma,mb)
            if spec is not f:
                np.testing.assert_array_equal(a.truth(self.y),b.truth(yy))
                np.testing.assert_array_equal(a.predictions(ma,ra,(60,25,5)),b.predictions(mb,rb,(60,25,5)))

    def test_permutation_repeats_selection_and_preserves_instance_labels(self):
        shuffled=permutation_labels(self.y,instance_blocks(self.y),np.random.default_rng(6))
        for block in instance_blocks(self.y):
            np.testing.assert_array_equal(np.sort(shuffled[block]),np.sort(self.y[block]))
        tiled=np.repeat(shuffled[:,None],20,axis=1)
        self.assertTrue(np.all(tiled==tiled[:,:1]))
        original=PreparedFold.fit;calls=[]
        def spy(fold,y,nclass):
            calls.append((tuple(fold.train_i),tuple(y[fold.train_i])))
            return original(fold,y,nclass)
        with patch.object(PreparedFold,'fit',spy):
            result=nested(self.x,shuffled,self.plan,sizes=(60,25,5),details=False)
        expected=len(self.plan)+sum(len(f['inner']) for f in self.plan)
        self.assertEqual(len(calls),expected);self.assertEqual(result['ranking_fits'],expected)

    def test_bootstrap_uses_images_not_trials(self):
        predictions=np.repeat(self.y[:,None],20,axis=1)
        predictions[::2]=(predictions[::2]+1)%5
        a=bootstrap(predictions,self.y,1000)
        # Replicating seed predictions changes no image-level uncertainty.
        b=bootstrap(np.repeat(predictions,3,axis=1),self.y,1000)
        self.assertEqual(a,b);self.assertEqual(a['number_of_instance_units'],25)

    def test_protocol_lock(self):
        lock=json.loads((RESULTS/'protocol-lock.json').read_text())
        self.assertEqual(lock['design_sha256'],digest((RESULTS/'design.md').read_bytes()))
        self.assertEqual(lock['manifest_sha256'],digest((RESULTS/'dataset-manifest.json').read_bytes()))


if __name__=='__main__':unittest.main()
