import inspect,tempfile,unittest
from pathlib import Path
import numpy as np
from flymon.exploration import ExplorationConfig,FrozenExplorationController,FrozenInterfaceController
from flymon.interaction import *
from flymon.motor import MOTOR_TYPES,resolve_dn_ids,resolve_motor_populations,verify_motor_neurons
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
  for cls in (BaselinePopulationDecoder,BaselineEventController,TransparentEventController,PopulationEventController,FrozenInterfaceController):
   self.assertEqual(list(inspect.signature(cls.decode).parameters),["self","rates"])
  src=Path("run_interface_experiment.py").read_text()+Path("run_dnp01_event_experiment.py").read_text()+Path("run_population_event_experiment.py").read_text()+Path("flymon/interaction.py").read_text()
  for forbidden in ("read_memory","memory[","player_position","map_id","reward","random.choice","desired_action"):self.assertNotIn(forbidden,src)
 def event_config(self,strategy,threshold=0,window=1):
  return EventDecoderConfig(strategy,"sum",2.,1.,threshold,window,.5,2)
 def cell(self,value):return {"DNp01_cells_hz":(value/2,value/2)}
 def test_reference_event_regression_exact(self):
  b=self.baseline();old=BaselineEventController(b,EventConfig("X",1,1,5,True))
  self.assertEqual([old.decode({"X":v})[0] for v in (1,3,3,1,3)],[None,"A",None,None,None])
 def test_transparent_rising_edge_matches_reference(self):
  baseline=FrozenPopulationBaseline({"DNp01":2.},{"DNp01":1.},{"DNp01":(1.,1.)},10)
  old=BaselineEventController(baseline,EventConfig("DNp01",1,1,0,True))
  new=TransparentEventController(EventDecoderConfig("rising_edge","sum",2.,1.,1.,1,.5,0))
  values=(2.,4.,3.25,2.9,4.)
  self.assertEqual([old.decode({"DNp01":v})[0] for v in values],[new.decode(self.cell(v))[0] for v in values])
 def test_level_decoder_threshold_and_sustained_no_spam(self):
  c=TransparentEventController(self.event_config("level",-.5))
  out=[c.decode(self.cell(v))[0] for v in (1,2,2,2,2)]
  self.assertEqual(out.count("A"),1)
 def test_recovery_decoder(self):
  c=TransparentEventController(self.event_config("recovery",1,3))
  out=[c.decode(self.cell(v))[0] for v in (0,0,0,0,3,4,5)]
  self.assertIn("A",out)
 def test_derivative_decoder(self):
  c=TransparentEventController(self.event_config("derivative",1,1))
  self.assertEqual([c.decode(self.cell(v))[0] for v in (1,1,4)][-1],"A")
 def test_cumulative_decoder_and_reset(self):
  c=TransparentEventController(self.event_config("cumulative",0,3))
  out=[c.decode(self.cell(v))[0] for v in (3,3,3)]
  self.assertEqual(out[-1],"A");self.assertEqual(len(c.history),0)
  self.assertIsNone(c.decode(self.cell(3))[0]);c.reset();self.assertEqual(len(c.history),0)
 def test_new_event_refractory_and_ablation(self):
  c=TransparentEventController(self.event_config("level",0),True)
  self.assertTrue(all(c.decode(self.cell(v))[0] is None for v in (4,0,4,0,4)))
  c=TransparentEventController(self.event_config("level",0))
  out=[c.decode(self.cell(v))[0] for v in (4,0,4,0,4)]
  self.assertEqual(out.count("A"),2);self.assertGreater(c.suppressions,0)
 def test_integrated_without_event_preserves_direction(self):
  sb,lb,cfg=self.controllers();direction=FrozenExplorationController(sb,lb,cfg,False)
  wrapped=FrozenInterfaceController(direction,None,None)
  rates={"DNa02_L":4,"DNa02_R":0,"DNg100":3,"MDN":0}
  self.assertEqual(wrapped.decode(rates)[0],"LEFT")
 def test_two_cell_aggregation_deterministic(self):
  self.assertEqual(aggregate_dnp01((2,4),"sum"),6);self.assertEqual(aggregate_dnp01((2,4),"mean"),3);self.assertEqual(aggregate_dnp01((2,4),"max"),4);self.assertEqual(aggregate_dnp01((2,4),"difference"),-2)

 def test_frozen_e4_population_artifact_and_id_resolution(self):
  historical=__import__("json").loads(Path("results/experiment-06-generalization/classification.json").read_text())["frozen_experiment4_ids"]
  archive=np.load("results/experiment-04-temporal-dn/dn-bins-5.npz")
  self.assertEqual(len(historical),20)
  self.assertEqual([x["flywire_id"] for x in historical],
                   [int(archive["flywire_ids"][x["dn_slot"]]) for x in historical])
  b=FakeBrain()
  with tempfile.TemporaryDirectory() as d:
   np.savez(Path(d)/"brain.npz",cell_type=b.cell_type,ids=np.arange(5)+100)
   meta,indices=resolve_dn_ids(b,Path(d),(104,100))
   self.assertEqual(indices.tolist(),[4,0])
   self.assertEqual([x["flywire_malecns_id"] for x in meta["neurons"]],[104,100])
 def test_vector_baseline_and_population_z(self):
  b=FrozenVectorBaseline((1.,2.),(0.,4.),10)
  np.testing.assert_allclose(population_z_vector((2.,4.),b,1.),(1.,1.))
  estimated=estimate_frozen_vector_baseline(((0,1),(2,3)),1.)
  self.assertEqual(estimated.means_hz,(1.,2.))
  self.assertEqual(estimated.variances_hz2,(2.,2.))
 def test_population_signal_families(self):
  history=[np.array((0.,0.)),np.array((1.,-1.))]
  self.assertEqual(population_signal(history,"mean"),0.)
  self.assertEqual(population_signal(history,"norm"),1.)
  self.assertEqual(population_signal(history,"change"),1.)
  window=[np.array((0.,0.)),np.array((0.,0.)),np.array((2.,0.)),np.array((2.,0.))]
  self.assertAlmostEqual(population_signal(window,"window_change",2),2**.5)
 def population_controller(self,signal="mean",rule="rising",threshold=.5,refractory=2):
  baseline=FrozenVectorBaseline((0.,0.),(0.,0.),10)
  config=PopulationEventConfig(signal,rule,threshold,1.,1,1,.25,refractory)
  return PopulationEventController(baseline,config)
 def test_population_event_threshold_refractory_and_reset(self):
  c=self.population_controller(refractory=0)
  self.assertIsNone(c.decode({"event_population_rates_hz":(0.,0.)})[0])
  self.assertEqual(c.decode({"event_population_rates_hz":(1.,1.)})[0],"A")
  self.assertIsNone(c.decode({"event_population_rates_hz":(1.,1.)})[0])
  c.reset();self.assertEqual(len(c.z_history),0);self.assertTrue(c.armed)
  level=self.population_controller(rule="level",refractory=2)
  actions=[level.decode({"event_population_rates_hz":(1.,1.)})[0] for _ in range(4)]
  self.assertEqual(actions,["A",None,None,"A"])
 def test_population_event_ablation(self):
  c=PopulationEventController(FrozenVectorBaseline((5.,5.),(0.,0.),10),
      PopulationEventConfig("norm","level",1.,.5),ablated=True)
  self.assertTrue(all(c.decode({"event_population_rates_hz":(0.,100.)})[0] is None for _ in range(10)))
 def test_population_decoder_receives_only_rates(self):
  first=self.population_controller(refractory=0)
  second=self.population_controller(refractory=0)
  rates={"event_population_rates_hz":(1.,1.)}
  self.assertEqual(first.decode(rates),second.decode(rates | {
      "framebuffer":np.full((2,2),255),"frame_hash":"known","pixels":999}))
  source=inspect.getsource(PopulationEventController.decode)
  for forbidden in ("framebuffer","frame_hash","pixels","memory"):
   self.assertNotIn(forbidden,source)
 def test_population_permutations_deterministic(self):
  x=np.arange(20,dtype=float).reshape(5,4)
  np.testing.assert_array_equal(permute_population_cells(x,13),permute_population_cells(x,13))
  np.testing.assert_array_equal(np.sort(permute_population_cells(x,13),axis=1),np.sort(x,axis=1))
  np.testing.assert_array_equal(permute_population_time(x,17),permute_population_time(x,17))
  self.assertEqual(sorted(map(tuple,permute_population_time(x,17))),sorted(map(tuple,x)))
 def test_population_a_event_first_and_absent_direction_regression(self):
  sb,lb,cfg=self.controllers();rates={"DNa02_L":4,"DNa02_R":0,"DNg100":3,"MDN":0,
      "event_population_rates_hz":(1.,1.)}
  direction=FrozenExplorationController(sb,lb,cfg,False)
  absent=FrozenInterfaceController(direction,None,None)
  self.assertEqual(absent.decode(rates)[0],"LEFT")
  direction=FrozenExplorationController(sb,lb,cfg,False)
  event=self.population_controller(refractory=0)
  self.assertEqual(FrozenInterfaceController(direction,None,event).decode(rates)[0],"A")

if __name__=="__main__":unittest.main()
