import torch
import torch.nn as nn

#--------------------------#
# LSTM Cell Implementation #
#--------------------------#
class LSTMCell(nn.Module):
    """
    Implementation of the original LSTM cell from Hochreiter & Schmidhuber (1997).
    Note: This version does NOT include a forget gate, Implementing the 
    Constant Error Carousel (CEC) as originally proposed.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Initialising the Parameters for gates: input/output/cell
        # Input-to-hidden weights
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))

        # Hidden-to-hidden (recurrent) weights
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))

        # Biases
        self.bias = nn.Parameter(torch.empty(3 * hidden_size))

        self.reset_params()


    def reset_params(self):
        """
        Initialises parameters to ensure stable gradient flow
        at the start of training.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                # Xavier (Glorot) initialisation
                ## Maintains variance of activations and gradients across layers
                ## Essential for preventing exploding/vanishing gradients in deep RNNs
                nn.init.xavier_uniform_(param)

            if 'bias' in name:
                # Initialises biases to zero to ensure gates start in a 'neutral' state
                nn.init.constant_(param, 0.0)


    def forward(self, x, init_states):
        """
        x shape: (batch_size, input_size)
        init_states: h_prev, c_prev shape: (batch_size, hidden_size)
        """
        h_prev, c_prev = init_states

        # 1. Compute all ih and hh gate components
        ## Resulting shapes: (batch_size, 3 * hidden_size)
        gates_ih = torch.matmul(x, self.weight_ih.t())
        gates_hh = torch.matmul(h_prev, self.weight_hh.t())

        # 2. Add the gate components and the bias together
        total_gates = gates_ih + gates_hh + self.bias.unsqueeze(0)

        # 3. Split into the 3 original gates
        ## i: input gate, g: candidate cell, o: output gate
        i_gate, g_gate, o_gate = total_gates.chunk(3, dim=1)

        # 4. Apply activations
        i = torch.sigmoid(i_gate)
        g = torch.tanh(g_gate)
        o = torch.sigmoid(o_gate)

        # 5. CEC (Constant Error Carousel) Update (NO FORGET GATE)
        c_next = c_prev + (i * g)

        # 6. Hidden state update
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

#---------------------------#
# LSTM Model Implementation #
#---------------------------#
class DeepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        # No of hidden layers
        self.num_layers = num_layers

        # ModuleList to track parameters of every layer
        self.lstm = nn.ModuleList()

        # Layer 1 takes raw input_size
        self.lstm.append(LSTMCell(input_size, hidden_size))

        # Subsequent layers take the hidden_size of previous layers as input
        for _ in range(1, num_layers):
            self.lstm.append(LSTMCell(hidden_size, hidden_size))

    def forward(self, x, states=None):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()

        # Initialise states for all layers: (num_layers, batch_size, hidden_size)
        if states is None:
            # Usage of x.new_zeros(): To inherit device and dtype automatically
            h_states = [x.new_zeros(batch_size, self.hidden_size)
                        for _ in range(self.num_layers)]
            c_states = [x.new_zeros(batch_size, self.hidden_size)
                        for _ in range(self.num_layers)]
        else:
            h_states, c_states = states
        
        seq_outs = []

        for t in range(seq_len):
            # Input for the first hidden layer
            current_in = x[:, t, :]

            new_h_states = []
            new_c_states = []

            # Pass the signal UP through the stack of layers
            for i, layer in enumerate(self.lstm):
                # The output h of layer i -> input x for layer i+1
                h_next, c_next = layer(current_in, (h_states[i], c_states[i]))

                new_h_states.append(h_next)
                new_c_states.append(c_next)

                # Update current input for the next layer in the stack
                current_in = h_next

            # Update the persistent states for the next time step
            h_states, c_states = new_h_states, new_c_states

            # Take h values from the very LAST layer as the seq. output
            # h_states[-1].unsqueeze(1): (batch_size, 1, hiddeen_size)
            seq_outs.append(h_states[-1].unsqueeze(1))

        # final outputs for each time step: (batch_size, T, hidden_size)
        final_outs = torch.cat(seq_outs, dim=1)

        # Return logits and h/c_states stacked into 3D Tensors
        return final_outs, (torch.stack(h_states), torch.stack(c_states))
    
#-----------------------------#
# LSTM Wrapper Implementation #
#-----------------------------#
class LSTMWrapper(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        ## Embedding layer
        self.embed = nn.Embedding(vocab_size, hidden_size)
        ## LSTM layers
        self.lstm = DeepLSTM(hidden_size, hidden_size, num_layers)
        ## Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, states=None):
        x = self.embed(x)
        out, states = self.lstm(x, states)
        logits = self.fc(out)

        return logits, states