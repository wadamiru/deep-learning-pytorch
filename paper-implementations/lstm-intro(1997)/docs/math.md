# LSTM Mathematical Formulation

This document outlines the mathematical formulation of the LSTM network
implemented in `src/lstm_cell.py`. The equations follow the original
Hochreiter & Schmidhuber (1997) paper, with PyTorch tensor variable mapping.

---

## Notation

- \( x_t \) : input at time step t  
- \( h_t \) : hidden state at time step t  
- \( c_t \) : cell state at time step t  
- \( W_*, U_*, b_* \) : learnable weights and biases for each gate  
- \( \sigma \) : sigmoid activation function  
- \( \tanh \) : hyperbolic tangent activation

---

## LSTM Cell Equations

### Input gate
\[
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
\]  
**Code mapping:** `i = torch.sigmoid(Wi @ x + Ui @ h_prev + bi)`

---

### Output gate
\[
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
\]  
**Code mapping:** `o = torch.sigmoid(Wo @ x + Uo @ h_prev + bo)`

---

### Cell candidate
\[
\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
\]  
**Code mapping:** `g = torch.tanh(Wc @ x + Uc @ h_prev + bc)`

---

### Cell state
\[
c_t = i_t \odot \tilde{c}_t + c_{t-1}  \quad \text{(Constant Error Carousel)}
\]  
**Code mapping:** `c = i * g + c_prev`

> Note: In the original 1997 formulation, the forget gate is not used.
> The previous cell state \( c_{t-1} \) flows directly (CEC).

---

### Hidden state
\[
h_t = o_t \odot \tanh(c_t)
\]  
**Code mapping:** `h = o * torch.tanh(c)`

---

## Sequence Unrolling

For a sequence of length T:

1. Initialize `h[-1]` and `c[-1]` (typically zeros)  
2. For t = 0 to T-1:  
   - Compute `i, o, g`  
   - Update cell state `c`  
   - Compute hidden state `h`  
3. Collect `h_t` for output or next layer

---

## Backpropagation Notes

- PyTorch **autograd** is used to compute gradients automatically.  
- The original paper uses **manual BPTT**, but here autograd handles it.  
- Mapping from equations to code ensures clarity and reproducibility.  

---

## References

- Hochreiter, S., Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
