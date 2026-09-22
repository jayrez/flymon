# flyvis positive control — Experiment 18 notes

**Outcome: F3** (cannot be made operational after a bounded attempt). This does not
become Experiment 18's verdict; the HR correlator (Experiment 17) and the in-experiment
`M_HR` control architecture serve as positive controls instead.

## What was attempted

1. **Isolated virtual environment** — `python3 -m venv` fails on this host because
   `ensurepip` is not installed, and creating a venv from the Flymon venv interpreter
   hits the same limitation. Falling back to `pip install --target /tmp/flyvis-libs`
   keeps flyvis fully outside both the repository and the Flymon environment.
2. **Install** — `flyvis==1.2.0` installs and imports cleanly once torch and
   torchvision come from the same CPU index (an initial ABI mismatch,
   `operator torchvision::nms does not exist`, came from torchvision resolving against
   PyPI while torch came from the PyTorch CPU index).
3. **Pretrained checkpoints** — flyvis installs **no console script**, so the
   documented `flyvis download-pretrained` command does not exist here. Searching the
   package finds only `download_sintel` / `download_url_to_file`, which fetch the
   Sintel *training* dataset, not trained networks. `flyvis.Ensemble("flow/0000")`
   raises `FileNotFoundError` on a local results directory that nothing populates.

## Why an untrained network was not substituted

flyvis T4/T5 direction selectivity emerges from task training. An untrained,
connectome-initialised network would not be expected to show it, so a null result
would say nothing about the stimuli and could easily be misread as evidence against
them. Reporting F3 honestly is preferable.

## Consequence for the interpretation matrix

Category **E** ("pretrained flyvis succeeds while the MaleCNS reconstruction fails")
could not be evaluated. It remains the most valuable follow-up: it would separate
"the MaleCNS neural-model abstraction is the limitation" from "this reconstruction is
the limitation". Obtaining the published checkpoints (or training an ensemble) is
recommended for the next experiment.

Nothing from flyvis — weights, caches, or the install directory — is committed.
