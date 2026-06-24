import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math

from functions import dirichlet_smooth

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.scalar = math.sqrt(84)

        self.AttentionLayerNorm = nn.LayerNorm(504)

        # Head 1
        self.Query_Matrix_1 = nn.Linear(504, 84, bias=False)
        self.Key_Matrix_1 = nn.Linear(504, 84, bias=False)
        self.Value_Matrix_1 = nn.Linear(504, 84, bias=False)

        # Head 2
        self.Query_Matrix_2 = nn.Linear(504, 84, bias=False)
        self.Key_Matrix_2 = nn.Linear(504, 84, bias=False)
        self.Value_Matrix_2 = nn.Linear(504, 84, bias=False)

        # Head 3
        self.Query_Matrix_3 = nn.Linear(504, 84, bias=False)
        self.Key_Matrix_3 = nn.Linear(504, 84, bias=False)
        self.Value_Matrix_3 = nn.Linear(504, 84, bias=False)

        # Head 4
        self.Query_Matrix_4 = nn.Linear(504, 84, bias=False)
        self.Key_Matrix_4 = nn.Linear(504, 84, bias=False)
        self.Value_Matrix_4 = nn.Linear(504, 84, bias=False)

        # Head 5
        self.Query_Matrix_5 = nn.Linear(504, 84, bias=False)
        self.Key_Matrix_5 = nn.Linear(504, 84, bias=False)
        self.Value_Matrix_5 = nn.Linear(504, 84, bias=False)

        # Head 6
        self.Query_Matrix_6 = nn.Linear(504, 84, bias=False)
        self.Key_Matrix_6 = nn.Linear(504, 84, bias=False)
        self.Value_Matrix_6 = nn.Linear(504, 84, bias=False)

        # Multi-Head Projection
        self.MultiHead_Matrix = nn.Linear(504, 504, bias=False)

        # MLP, Post Attention
        self.Layer_1 = nn.Linear(504, 504)
        self.MLPLayerNorm = nn.LayerNorm(504)
        self.Layer_2 = nn.Linear(504, 504) # ReLU for Activation
        

    def forward(self, embeddings): # Input of k+1 ROWS by 504 column. First k are memories, last column is central query.
        # In general, always keep rows as individual embeddings 

        average_attention = torch.zeros(7,7)

        res = embeddings
        embeddings = self.AttentionLayerNorm(embeddings)

        # Head 1
        queries = self.Query_Matrix_1(embeddings)
        keys = self.Key_Matrix_1(embeddings)
        values = self.Value_Matrix_1(embeddings)

        dots = queries @ keys.T
        dots = dots / self.scalar
        dots[:, -1] = -torch.inf # Mask

        attention = F.softmax(dots,dim=1) # Each row corresponds to a query with d_k Keys
        average_attention += attention

        deltas_1 = attention @ values # Each row gives value change for a particular query 

        # Head 2
        queries = self.Query_Matrix_2(embeddings)
        keys = self.Key_Matrix_2(embeddings)
        values = self.Value_Matrix_2(embeddings)

        dots = queries @ keys.T
        dots = dots / self.scalar
        dots[:, -1] = -torch.inf # Mask

        attention = F.softmax(dots,dim=1) # Each row corresponds to a query with d_k Keys
        average_attention += attention

        deltas_2 = attention @ values # Each row gives value change for a particular query 

        # Head 3
        queries = self.Query_Matrix_3(embeddings)
        keys = self.Key_Matrix_3(embeddings)
        values = self.Value_Matrix_3(embeddings)

        dots = queries @ keys.T
        dots = dots / self.scalar
        dots[:, -1] = -torch.inf # Mask

        attention = F.softmax(dots,dim=1) # Each row corresponds to a query with d_k Keys
        average_attention += attention

        deltas_3 = attention @ values # Each row gives value change for a particular query 

        # Head 4
        queries = self.Query_Matrix_4(embeddings)
        keys = self.Key_Matrix_4(embeddings)
        values = self.Value_Matrix_4(embeddings)

        dots = queries @ keys.T
        dots = dots / self.scalar
        dots[:, -1] = -torch.inf # Mask

        attention = F.softmax(dots,dim=1) # Each row corresponds to a query with d_k Keys
        average_attention += attention

        deltas_4 = attention @ values # Each row gives value change for a particular query 

        # Head 5
        queries = self.Query_Matrix_5(embeddings)
        keys = self.Key_Matrix_5(embeddings)
        values = self.Value_Matrix_5(embeddings)

        dots = queries @ keys.T
        dots = dots / self.scalar
        dots[:, -1] = -torch.inf # Mask

        attention = F.softmax(dots,dim=1) # Each row corresponds to a query with d_k Keys
        average_attention += attention

        deltas_5 = attention @ values # Each row gives value change for a particular query 

        # Head 6
        queries = self.Query_Matrix_6(embeddings)
        keys = self.Key_Matrix_6(embeddings)
        values = self.Value_Matrix_6(embeddings)

        dots = queries @ keys.T
        dots = dots / self.scalar
        dots[:, -1] = -torch.inf # Mask

        attention = F.softmax(dots,dim=1) # Each row corresponds to a query with d_k Keys
        average_attention += attention

        deltas_6 = attention @ values # Each row gives value change for a particular query 

        # Concatenate and Project
        tensors = (deltas_1, deltas_2, deltas_3, deltas_4, deltas_5, deltas_6)
        concatenation = torch.cat(tensors=tensors, dim=1)

        deltas = self.MultiHead_Matrix(concatenation)

        # Update Residuals
        res = res + deltas

        # MLP Block
        res_2 = res
        res = self.MLPLayerNorm(res)
        res = self.Layer_1(res)
        res = F.relu(res)
        res = self.Layer_2(res)
        res_2 = res_2 + res

        return res_2, average_attention / 6


# ------------------+

class MemoryTransformer(nn.Module):

    def __init__(self, k):
        super().__init__()

        self.k = k

        # Encoder ----+
        self.Encoder = nn.Linear(262, 504, bias=False)

        # Main Blocks ----+
        self.Block1 = TransformerBlock()
        self.Block2 = TransformerBlock()
        self.Block3 = TransformerBlock()
        self.Block4 = TransformerBlock()
        self.Block5 = TransformerBlock()
        self.Block6 = TransformerBlock()

        # Decoder ----+
        self.Decoder = nn.Linear(504, 262, bias=False)

    def forward(self, input): # Input of (k+1) x (252+7+1+2)

        # Work in Latent Space
        visits_transformed = torch.log(dirichlet_smooth(input[:, 252:259]))
        value_clamped = torch.clamp(input[:, 259:260], -1 + 1e-6, 1 - 1e-6)
        value_transformed = 2 * torch.atanh(value_clamped)
        input = torch.cat([input[:, :252], visits_transformed, value_transformed, input[:, 260:262]], dim=1)

        embeddings = self.Encoder(input)

        embeddings, average_attention_1 = self.Block1(embeddings)
        embeddings, average_attention_2 = self.Block2(embeddings)
        embeddings, average_attention_3 = self.Block3(embeddings)
        embeddings, average_attention_4 = self.Block4(embeddings)
        embeddings, average_attention_5 = self.Block5(embeddings)
        embeddings, average_attention_6 = self.Block6(embeddings)

        average_attention = (average_attention_1 + average_attention_2 + average_attention_3 + average_attention_4 + average_attention_5 + average_attention_6) / 6

        decodings = self.Decoder(embeddings)

        expected_child_CNN_embeddings_decoded = F.relu(decodings[:, :252])
        visits_decoded = F.softmax(decodings[:, 252:259], dim=1)
        value_decoded = 2 * torch.sigmoid(decodings[:, 259:260]) - 1
        roles_decoded = F.hardsigmoid(decodings[:, 260:262])

        decodings = torch.cat([expected_child_CNN_embeddings_decoded, visits_decoded, value_decoded, roles_decoded], dim=1)

        return decodings, average_attention
    

    def fit(self, training_inputs, training_outputs, epochs, nabla, batch_size=40):

        memory_expected_child_mse_losses = []
        memory_cross_entropy_losses = []
        memory_value_mse_losses = []
        memory_role_mse_losses = []

        centre_expected_child_mse_losses = []
        centre_cross_entropy_losses = []
        centre_value_mse_losses = []
        centre_role_mse_losses = []

        # Batches as Tensors (Lists of Matrices)
        # Role-Corrected Inputs 

        training_size = training_inputs.shape[0]
        mse = nn.MSELoss()
        optimiser = optim.Adam(self.parameters(), lr=nabla)

        for epoch in range(epochs):
            permutation = torch.randperm(training_size)
            training_inputs = training_inputs[permutation]
            training_outputs = training_outputs[permutation]

            for i in range(0, training_size, batch_size):
                batch_inputs = training_inputs[i: i + batch_size]
                batch_outputs = training_outputs[i: i + batch_size]

                optimiser.zero_grad()
                loss_scalar = torch.tensor(0.0)

                # Regularisation

                L2 = 0
                for param in self.parameters():
                    L2 += torch.sum(param ** 2)

                for x, y in zip(batch_inputs, batch_outputs):

                    transformer_output, _ = self.forward(x)

                    k_memory_outputs = transformer_output[:-1]
                    centre_output = transformer_output[-1]

                    true_k_memories = y[:-1]
                    true_centre = y[-1]

                    # Compute Loss for k Memories

                    memory_expected_child_mse = ((k_memory_outputs[:, :252] - true_k_memories[:, :252])**2).sum() / 252
                    clone = memory_expected_child_mse
                    memory_expected_child_mse_losses.append((clone.detach()).item())

                    memory_cross_entropy = -(k_memory_outputs[:, 252:259] * torch.log(true_k_memories[:, 252:259] + 1e-8)).sum()
                    clone = memory_cross_entropy
                    memory_cross_entropy_losses.append((clone.detach()).item())

                    memory_value_mse = ((k_memory_outputs[:, 259] - true_k_memories[:, 259])**2).sum()
                    clone = memory_value_mse
                    memory_value_mse_losses.append((clone.detach()).item())

                    memory_role_mse = ((k_memory_outputs[:, 260:262] - true_k_memories[:, 260:262])**2).sum()
                    clone = memory_role_mse
                    memory_role_mse_losses.append((clone.detach()).item())

                    memory_loss = (memory_expected_child_mse + memory_cross_entropy + memory_value_mse + memory_role_mse) / self.k

                    # Compute Loss for Centre

                    centre_expected_child_mse = ((centre_output[:252] - true_centre[:252])**2).sum() / 252
                    clone = centre_expected_child_mse
                    centre_expected_child_mse_losses.append((clone.detach()).item())

                    centre_cross_entropy = -torch.dot(centre_output[252:259], torch.log(true_centre[252:259] + 1e-8))
                    clone = centre_cross_entropy
                    centre_cross_entropy_losses.append((clone.detach()).item())

                    centre_value_mse = (centre_output[259] - true_centre[259])**2
                    clone = centre_value_mse
                    centre_value_mse_losses.append((clone.detach()).item())

                    centre_role_mse = ((centre_output[260:262] - true_centre[260:262])**2).sum()
                    clone = centre_role_mse
                    centre_role_mse_losses.append((clone.detach()).item())

                    centre_loss = centre_expected_child_mse + centre_cross_entropy + centre_value_mse + centre_role_mse
                    loss_scalar += memory_loss + centre_loss + 1e-4 * L2

                loss_scalar.backward()
                optimiser.step()

        return memory_expected_child_mse_losses, memory_cross_entropy_losses, memory_value_mse_losses, memory_role_mse_losses, centre_expected_child_mse_losses, centre_cross_entropy_losses, centre_value_mse_losses, centre_role_mse_losses

    def test(self, test_inputs, test_outputs, hidden_indices):
        totals = {
            "memory_child_mse": 0,
            "memory_cross_entropy": 0,
            "memory_value_mse": 0,
            "memory_role_mse": 0,
            "centre_child_mse": 0,
            "centre_cross_entropy": 0,
            "centre_value_mse": 0,
            "centre_role_mse": 0
        }

        n = len(test_inputs)
        attention_accumulator = None

        with torch.no_grad():
            for x, y, hidden_index in zip(test_inputs, test_outputs, hidden_indices):

                transformer_output, attention = self.forward(x)
                attention = attention[-1, hidden_index]

                # Accumulate attention
                if attention_accumulator is None:
                    attention_accumulator = attention
                else:
                    attention_accumulator += attention

                k_memory_outputs = transformer_output[:-1]
                centre_output = transformer_output[-1]
                true_k_memories = y[:-1]
                true_centre = y[-1]

                totals["memory_child_mse"]     += (((k_memory_outputs[:, :252] - true_k_memories[:, :252])**2).sum() / 252).item()
                totals["memory_cross_entropy"] += (-(k_memory_outputs[:, 252:259] * torch.log(true_k_memories[:, 252:259] + 1e-8)).sum()).item()
                totals["memory_value_mse"]     += (((k_memory_outputs[:, 259] - true_k_memories[:, 259])**2).sum()).item()
                totals["memory_role_mse"]      += (((k_memory_outputs[:, 260:262] - true_k_memories[:, 260:262])**2).sum()).item()
                totals["centre_child_mse"]     += (((centre_output[:252] - true_centre[:252])**2).sum() / 252).item()
                totals["centre_cross_entropy"] += (-torch.dot(centre_output[252:259], torch.log(true_centre[252:259] + 1e-8))).item()
                totals["centre_value_mse"]     += ((centre_output[259] - true_centre[259])**2).item()
                totals["centre_role_mse"]      += (((centre_output[260:262] - true_centre[260:262])**2).sum()).item()

        average_attention = attention_accumulator / n

        return {k: v / n for k, v in totals.items()}, average_attention