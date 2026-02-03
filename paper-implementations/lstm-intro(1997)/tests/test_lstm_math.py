import torch
import torch.nn as nn
from src.lstm_cell import LSTMCell1997 # Assuming this is your class name

def test_manual_vs_pytorch():
    input_size = 10
    hidden_size = 20
    batch_size = 1
    
    # 1. Initialize your manual cell
    manual_cell = LSTMCell1997(input_size, hidden_size)
    
    # 2. Initialize PyTorch's official LSTM
    # We use a single layer and a single step
    pt_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    
    # 3. CRITICAL: Synchronize Weights
    # PyTorch stores weights as [W_ii|W_if|W_ig|W_io] (concatenated)
    with torch.no_grad():
        # Copy your weights into the PT model
        # Indexing: input gate(0), forget gate(1), cell candidate(2), output gate(3)
        
        # We set Forget Gate (index 1) to be "always on" (identity)
        # to match your 1997 implementation (CEC)
        pt_lstm.weight_ih_l0[hidden_size:2*hidden_size].fill_(0)
        pt_lstm.weight_hh_l0[hidden_size:2*hidden_size].fill_(0)
        pt_lstm.bias_ih_l0[hidden_size:2*hidden_size].fill_(100) # High bias = Sigmoid(1)
        pt_lstm.bias_hh_l0[hidden_size:2*hidden_size].fill_(100)

        # Copy Input Gate weights
        pt_lstm.weight_ih_l0[0:hidden_size] = manual_cell.Wi
        pt_lstm.weight_hh_l0[0:hidden_size] = manual_cell.Ui
        pt_lstm.bias_ih_l0[0:hidden_size] = manual_cell.bi
        
        # Copy Cell Candidate weights (often called 'g' in PT)
        pt_lstm.weight_ih_l0[2*hidden_size:3*hidden_size] = manual_cell.Wc
        pt_lstm.weight_hh_l0[2*hidden_size:3*hidden_size] = manual_cell.Uc
        pt_lstm.bias_ih_l0[2*hidden_size:3*hidden_size] = manual_cell.bc

        # Copy Output Gate weights
        pt_lstm.weight_ih_l0[3*hidden_size:] = manual_cell.Wo
        pt_lstm.weight_hh_l0[3*hidden_size:] = manual_cell.Uo
        pt_lstm.bias_ih_l0[3*hidden_size:] = manual_cell.bo

    # 4. Create dummy input
    x = torch.randn(batch_size, 1, input_size)
    h_init = torch.zeros(1, batch_size, hidden_size)
    c_init = torch.zeros(1, batch_size, hidden_size)

    # 5. Run both
    manual_h, manual_c = manual_cell(x[0], (h_init[0], c_init[0]))
    pt_out, (pt_h, pt_c) = pt_lstm(x, (h_init, c_init))

    # 6. Assertion
    # atol=1e-6 accounts for minor floating point differences in hardware
    assert torch.allclose(manual_h, pt_h[0], atol=1e-6), "Hidden state mismatch!"
    assert torch.allclose(manual_c, pt_c[0], atol=1e-6), "Cell state mismatch!"
    
    print("Math Verification Passed: Manual implementation matches PyTorch (1997 mode).")

if __name__ == "__main__":
    test_manual_vs_pytorch()
