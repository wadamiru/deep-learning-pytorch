# LSTM Mathematical Formulation

This document outlines the mathematical formulation of the LSTM network
implemented in `src/lstm_cell.py`. The equations follow the original
Hochreiter & Schmidhuber (1997) paper, with PyTorch tensor variable mapping.

---

## Notation

- $x_t$ : input at time step $t$  
- $h_t$ : hidden state at time step $t$ 
- $c_t$ : cell state at time step $t$  
- $W_\*, U_\*, b_\*$ : learnable weights and biases for each gate  
- $\sigma$ : sigmoid activation function  
- $\tanh$ : hyperbolic tangent activation

---

## LSTM Cell Equations

### Input gate

$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$

**Code mapping:** `i = torch.sigmoid(Wi @ x + Ui @ h_prev + bi)`

---

### Output gate

$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$

**Code mapping:** `o = torch.sigmoid(Wo @ x + Uo @ h_prev + bo)`

---

### Cell candidate

$$\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
  
**Code mapping:** `g = torch.tanh(Wc @ x + Uc @ h_prev + bc)`

---

### Cell state

$$c_t = i_t \odot \tilde{c}_t + c_{t-1}  \quad \text{(Constant Error Carousel)}$$
 
**Code mapping:** `c = i * g + c_prev`

> Note: In the original 1997 formulation, the forget gate is not used.
> The previous cell state $c_{t-1}$ flows directly (CEC).

---

### Hidden state

$$h_t = o_t \odot \tanh(c_t)$$
 
**Code mapping:** `h = o * torch.tanh(c)`

---

## The "Gradient Highway": Solving Vanishing Gradients

The primary contribution of the 1997 LSTM is the **Constant Error Carousel (CEC)**. 

### The Problem in Vanilla RNNs
In a standard RNN, the gradient $\frac{\partial h_k}{\partial h_t}$ involves a product of $k-t$ weight matrices ($W_{hh}$). If the weights are small, the gradient vanishes exponentially:

$$\frac{\partial h_k}{\partial h_t} \propto (W_{hh})^{k-t}$$

### The LSTM Solution
In this implementation, the derivative of the cell state $c_t$ with respect to the previous state $c_{t-1}$ is:

$$\frac{\partial c_t}{\partial c_{t-1}} = \frac{\partial}{\partial c_{t-1}} [i_t \odot \tilde{c}_t + c_{t-1}] = 1$$

Because this derivative is **1**, the gradient can flow through hundreds of time steps without shrinking to zero, rather constant at 1.

$$
\frac{\partial c_k}{\partial c_t}
=
\prod_{j=t+1}^k \frac{\partial c_j}{\partial c_{j-1}}
=
\prod_{j=t+1}^k 1
=
1
$$

This "Identity Mapping" is mathematically similar to the Skip Connections later used in ResNets.

---

## Sequence Unrolling

For a sequence of length $T$:

1. Initialize `h[-1]` and `c[-1]` (typically zeros)  
2. For $t = 0$ to $T-1$:  
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
