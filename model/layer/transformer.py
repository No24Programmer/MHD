
import torch
import torch.nn as nn
import numpy as np

def truncated_normal(t, mean=0.0, std=0.01):
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
        cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
        if not torch.sum(cond):
            break
        t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
    return t


class Multi_Head_Attention_For_Dynamic(torch.nn.Module):
    def __init__(self, d_key, d_value, d_model, n_head, attention_dropout):
        super(Multi_Head_Attention_For_Dynamic, self).__init__()
        self.d_key = d_key
        self.d_query = d_value
        self.d_value = d_value
        self.d_model = d_model
        self.n_head = n_head
        self.attention_dropout = attention_dropout

        self.q_n = torch.nn.Linear(self.d_model, self.d_key * self.n_head)
        self.q_n.weight.data = truncated_normal(self.q_n.weight.data, std=0.02)
        torch.nn.init.constant_(self.q_n.bias, 0.0)
        self.k_n = torch.nn.Linear(self.d_model, self.d_key * self.n_head)
        self.k_n.weight.data = truncated_normal(self.k_n.weight.data, std=0.02)
        torch.nn.init.constant_(self.k_n.bias, 0.0)
        self.v_n = torch.nn.Linear(self.d_model, self.d_key * self.n_head)
        self.v_n.weight.data = truncated_normal(self.v_n.weight.data, std=0.02)
        torch.nn.init.constant_(self.v_n.bias, 0.0)

        self.q_e = torch.nn.Linear(self.d_model, self.d_key * self.n_head)
        self.q_e.weight.data = truncated_normal(self.q_e.weight.data, std=0.02)
        torch.nn.init.constant_(self.q_e.bias, 0.0)
        self.k_e = torch.nn.Linear(self.d_model, self.d_key * self.n_head)
        self.k_e.weight.data = truncated_normal(self.k_e.weight.data, std=0.02)
        torch.nn.init.constant_(self.k_e.bias, 0.0)
        self.v_e = torch.nn.Linear(self.d_model, self.d_key * self.n_head)
        self.v_e.weight.data = truncated_normal(self.v_e.weight.data, std=0.02)
        torch.nn.init.constant_(self.v_e.bias, 0.0)

        self.project_layer = torch.nn.Linear(d_value * n_head, self.d_model)
        self.project_layer.weight.data = truncated_normal(self.project_layer.weight.data, std=0.02)
        torch.nn.init.constant_(self.project_layer.bias, 0.0)
        self.project_e_layer = torch.nn.Linear(d_value * n_head, self.d_model)
        self.project_e_layer.weight.data = truncated_normal(self.project_e_layer.weight.data, std=0.02)
        torch.nn.init.constant_(self.project_e_layer.bias, 0.0)

        self.mlp = nn.Sequential(
            torch.nn.Linear(self.d_model, self.d_model),
            # torch.nn.ReLU(inplace=True),
            torch.nn.GELU(),
            torch.nn.Linear(self.d_model, self.d_model)
        )

        self.layer_norm1 = nn.LayerNorm(normalized_shape=self.d_model, eps=1e-7, elementwise_affine=True)

        self.activation_function = nn.GELU()

    def forward(self, X, incidence_matrix, attn_bias, E):
        # B is batch_size, N is max_node_len, H is n_head, D is d_key, M is max_hyperedge_number
        batch_size = X.size(0)
        # X = self.X_norm_layer(X)
        # E = self.E_norm_layer(E)

        # X is [B,N,H*D],  E is [B,M,H*D],
        q_n = self.q_n(X).view(batch_size, -1, self.n_head, self.d_query).transpose(1, 2)  # [B, H, N, D]
        k_n = self.k_n(X).view(batch_size, -1, self.n_head, self.d_key).transpose(1, 2)    # [B, H, N, D]
        v_n = self.v_n(X).view(batch_size, -1, self.n_head, self.d_value).transpose(1, 2)  # [B, H, N, D]

        q_e = self.q_e(E).view(batch_size, -1, self.n_head, self.d_query).transpose(1, 2)  # [B, H, M, D]

        att_e = torch.matmul(q_e, k_n.transpose(-1, -2)) / np.sqrt(self.d_key)
        zero_vec = -9e15 * torch.ones_like(att_e)
        att_e = torch.where(incidence_matrix.unsqueeze(dim=1).transpose(-2, -1) > 0, att_e, zero_vec)
        # weight_e = torch.nn.Dropout(self.attention_dropout)(torch.nn.Softmax(dim=-1)(att_e))  # [B, H, M, N]
        weight_e = torch.nn.Softmax(dim=-1)(att_e)  # [B, H, M, N]
        E_ = self.activation_function(weight_e.matmul(v_n))  # [B, H, M, D]

        E_ = E_.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_value)
        # E_ = self.project_e_layer(E_)  # [B, M, H*D]

        # E_ = self.mlp(torch.cat((E, E_), dim=-1))
        E_ = self.mlp(self.layer_norm1(torch.add(E, E_)))

        k_e = self.k_e(E_).view(batch_size, -1, self.n_head, self.d_key).transpose(1, 2)  # [B, H, M, D]
        v_e = self.v_e(E_).view(batch_size, -1, self.n_head, self.d_key).transpose(1, 2)  # [B, H, M, D]

        att_n = torch.matmul(q_n, k_e.transpose(-1, -2)) / np.sqrt(self.d_key)
        zero_vec = -9e15 * torch.ones_like(att_n)
        att_n = torch.where(incidence_matrix.unsqueeze(dim=1) > 0, att_n, zero_vec)
        # weight_n = torch.nn.Dropout(self.attention_dropout)(torch.nn.Softmax(dim=-1)(att_n))  # [B, H, N, M]
        weight_n = torch.nn.Softmax(dim=-1)(att_n)  # [B, H, N, M]
        X_ = self.activation_function(weight_n.matmul(v_e))  # [B, H, N, D]

        X_ = X_.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_value)
        # X_ = self.project_layer(X_)  # [B, N, H*D]

        return X_, E_





class Positionwise_Feed_Forward(torch.nn.Module):
    def __init__(self, d_inner_hid, d_model):
        super(Positionwise_Feed_Forward, self).__init__()
        self.d_inner_hid = d_inner_hid
        self.d_hid = d_model

        self.fc1 = torch.nn.Linear(self.d_hid, self.d_inner_hid)
        self.fc1.weight.data = truncated_normal(self.fc1.weight.data, std=0.02)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.constant_(self.fc1.bias, 0.0)
        self.fc2 = torch.nn.Linear(self.d_inner_hid, self.d_hid)
        self.fc2.weight.data = truncated_normal(self.fc2.weight.data, std=0.02)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        return self.fc2(torch.nn.GELU()(self.fc1(x)))


class encoder_layer(nn.Module):
    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout):
        super(encoder_layer, self).__init__()
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.d_inner_hid = d_inner_hid
        self.prepostprocess_dropout = prepostprocess_dropout
        self.attention_dropout = attention_dropout

        self.multi_head_attention = Multi_Head_Attention_For_Dynamic(
            self.d_key,
            self.d_value,
            self.d_model,
            self.n_head,
            self.attention_dropout)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=self.d_model, eps=1e-7, elementwise_affine=True)
        self.positionwise_feed_forward = Positionwise_Feed_Forward(self.d_inner_hid, self.d_model)
        self.layer_norm2 = torch.nn.LayerNorm(normalized_shape=self.d_model, eps=1e-7, elementwise_affine=True)

    def forward(self, enc_input, adj_matrix, attn_bias, e=None):
        attn_output, e_output = self.multi_head_attention(
            enc_input,
            adj_matrix,
            attn_bias,
            e)
        attn_output = self.layer_norm1(torch.add(enc_input, torch.nn.Dropout(self.prepostprocess_dropout)(attn_output)))
        ffd_output = self.positionwise_feed_forward(attn_output)
        ffd_output = self.layer_norm2(torch.add(attn_output, torch.nn.Dropout(self.prepostprocess_dropout)(ffd_output)))
        return ffd_output, e_output


class transformer_encoder(torch.nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_inner_hid, prepostprocess_dropout, attention_dropout):
        super(transformer_encoder, self).__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_key = int(d_model / n_head)
        self.d_value = int(d_model / n_head)
        self.d_model = d_model
        self.d_inner_hid = d_inner_hid
        self.prepostprocess_dropout = prepostprocess_dropout
        self.attention_dropout = attention_dropout

        for nl in range(self.n_layer):
            setattr(self, "encoder_layer{}".format(nl), encoder_layer(
                self.n_head,
                self.d_key,
                self.d_value,
                self.d_model,
                self.d_inner_hid,
                self.prepostprocess_dropout,
                self.attention_dropout))

    def forward(self, enc_input, adj_matrix, attn_bias=None, e=None):
        for nl in range(self.n_layer):
            enc_output, e_output = getattr(self, "encoder_layer{}".format(nl))(
                enc_input,
                adj_matrix,
                attn_bias,
                e
            )
            enc_input = enc_output
            e = e_output
        return enc_output, e




