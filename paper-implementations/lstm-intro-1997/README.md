# LSTM from the 1997 Paper

This repository implements the Long Short-Term Memory (LSTM) network as introduced by Hochreiter & Schmidhuber (1997), using Pytorch tensor operations without relying on `torch.nn.LSTM`.

Modern frameworks provide highly optimized LSTM implementations, but their
internal mechanics are hidden. This project focuses on implementing the LSTM
cell equations explicitly in code in order to:
- understand the original formulation from the paper,
- make the forward computation transparent,
- and demonstrate control over recurrent model internals.

## Reference
Hochreiter, S., Schmidhuber, J. (1997).
"Long Short-Term Memory", Neural Computation.

## Implementation highlights
- Explicit implementation of LSTM gate equations
- Manual sequence unrolling
- Backpropagation Through Time
- Uses Pytorch `tensors` and `autograd`
- No use of `torch.nn.LSTM` or `torch.nn.RNN`

## Notes on formulation
This implementation follows the structure described in the original 1997 paper.
Any deviations from the original formulation (e.g. inclusion of a forget gate)
are documented in the code and math notes.

## Project structure
```.
├── src/
│   ├── lstm.py
│   ├── train.py      # training logic only
│   ├── dataset.py       # dataset/ toy task
│   └── main.py       # entry point
├── docs/
│   └── math.md
│   └── benchmarks.md
├── data/
│   ├── full.txt
│   ├── train.txt      # training logic only
│   ├── val.txt       # dataset/ toy task
│   └── test.txt 
└── README.md
```

## Usage
Install dependencies:
```bash
pip install torch
```

## Run training
Use the entry point:
```bash
python src/main.py
```

This will:
- Prepare data (from `dataset.py`)
- Initialize the LSTM model (`lstm.py`)
- Run the training loop (`train.py`)

## Documentation
The full mathematical formulation of the forward pass and the mapping
from equations to code variables is provided in:

`docs/math.md`