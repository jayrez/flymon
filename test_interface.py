import inspect,tempfile,unittest
from pathlib import Path
import numpy as np
from flymon.exploration import ExplorationConfig,FrozenExplorationController,FrozenInterfaceController
from flymon.interaction import *
from flymon.motor import MOTOR_TYPES,resolve_motor_populations,verify_motor_neurons
from flymon.steering import SteeringBaseline
from flymon.locomotion import PopulationBaseline

class FakeBrain:
 def __init__(self):
  self.cell_type=np.array(["DNa02","DNg100","MDN","DNp01","ExtraDN"]);self.superclass=np.array(["descending_neuron"]*5);self.side=np.array(["L","R","L","R","L"])
 def cells(self,names):return np.flatnonzero(np.isin(self.cell_type,names))

class InterfaceTests(unittest.TestCase):
 def baseline(self):
  return FrozenPopulationBaseline({"X":1.},{"X":1.},{"X":(1.,)},10)
 def test_metadata_resolver_and_historical_wrapper(self):
  b=FakeBrain()
  with tempfile.TemporaryDirectory() as d:
   np.savez(Path(d)/"brain.npz",cell_type=b.cell_type,ids=np.arange(5)+100)
   old,op=verify_motor_neurons(b,Path(d));new,npop=resolve_motor_populations(b,Path(d),MOTOR_TYPES)
   self.assertEqual(old,new)
   for k in MOTOR_TYPES:np.testing.assert_array_equal(op[k],npop[k])
   extra,p=resolve_motor_populations(b,Path(d),("ExtraDN",));self.assertEqual(extra["counts"]["ExtraDN"],1)
 def test_baseline_is_frozen_and_threshold_ablation(self):
  b=self.baseline();before=asdict(b);d=BaselinePopulationDecoder(b,PopulationThresholdConfig("X",1,1,0),"DOWN")
  self.assertIsNone(d.decode({"X":2.})[0]);self.assertEqual(d.decode({"X":2.01})[0],"DOWN");self.assertEqual(asdict(b),before)
  self.assertIsNone(BaselinePopulationDecoder(b,PopulationThresholdConfig("X",1,1,0),"DOWN",True).decode({"X":99})[0])
 def test_event_edge_refractory_no_spam_and_ablation(self):
  c=BaselineEventController(self.baseline(),EventConfig("X",1,1,5,True))
  actions=[c.decode({"X":v})[0] for v in (1,3,3,3,1,3,1,3)]
  self.assertEqual(actions.count("A"),2);self.assertEqual(actions[:4], [None,"A",None,None])
  a=BaselineEventController(self.baseline(),EventConfig("X",1,1,5,True),True)
  self.assertTrue(all(a.decode({"X":v})[0] is None for v in (1,9,1,9)))
 def controllers(self):
  sb=SteeringBaseline(0,0,1,1,0,1,10);lb=PopulationBaseline(0,1,0,1,10,(0,),(0,))
  cfg=ExplorationConfig(2,1,"L",2,1,2,1,1);return sb,lb,cfg
 def test_experiment10_arbitration_unchanged(self):
  sb,lb,cfg=self.controllers();c=FrozenExplorationController(sb,lb,cfg,False)
  self.assertEqual(c.decode({"DNa02_L":4,"DNa02_R":0,"DNg100":3,"MDN":99})[0],"LEFT")
  c=FrozenExplorationController(sb,lb,cfg,True)
  self.assertEqual(c.decode({"DNa02_L":0,"DNa02_R":0,"DNg100":0,"MDN":5})[0],"DOWN")
 def test_integrated_a_priority_and_down(self):
  sb,lb,cfg=self.controllers();b=self.baseline()
  direction=FrozenExplorationController(sb,lb,cfg,False);down=BaselinePopulationDecoder(b,PopulationThresholdConfig("X",1,1,0),"DOWN");event=BaselineEventController(b,EventConfig("X",3,1,5,True))
  c=FrozenInterfaceController(direction,down,event)
  self.assertEqual(c.decode({"DNa02_L":0,"DNa02_R":0,"DNg100":0,"MDN":0,"X":3})[0],"DOWN")
  direction=FrozenExplorationController(sb,lb,cfg,False);event=BaselineEventController(b,EventConfig("X",1,1,5,True))
  self.assertEqual(FrozenInterfaceController(direction,down,event).decode({"DNa02_L":9,"DNa02_R":0,"DNg100":0,"MDN":0,"X":9})[0],"A")
 def test_controller_surface_and_no_ram(self):
  for cls in (BaselinePopulationDecoder,BaselineEventController,FrozenInterfaceController):
   self.assertEqual(list(inspect.signature(cls.decode).parameters),["self","rates"])
  src=Path("run_interface_experiment.py").read_text()+Path("flymon/interaction.py").read_text()
  for forbidden in ("read_memory","player_position","map_id","reward","random.choice","desired_action"):self.assertNotIn(forbidden,src)
if __name__=="__main__":unittest.main()
