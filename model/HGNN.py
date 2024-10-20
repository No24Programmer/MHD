import torch.nn

from model.layer.Loss import ComputeLoss
from model.layer.MLP import MLP
from model.layer.MVEncoder import MVEncoder
from model.layer.Measure_F import Measure_F
from model.layer.PositionalEncoding import PositionalEncoding


class HGNNModel(torch.nn.Module):
    def __init__(self, config, device=None):
        super(HGNNModel, self).__init__()
        self._emb_size = config['hidden_size']
        self._voc_size = config['vocab_size']
        self._n_relation = config['num_relations']
        self._e_soft_label = config['entity_soft_label']
        self._r_soft_label = config['relation_soft_label']
        self._use_dynamic = config['use_dynamic']
        self._device = device
        self._batch_size = config["batch_size"]
        self._num_view = config["num_view"]
        self._view_id = config["view_id"]
        self._loss_lamda = config["loss_lamda"]
        self._phi_num_layers = 2
        self._loss_mat = config["loss_mat"]
        self._loss_cor = config["loss_cor"]
        self._use_disentangled = config["use_disentangled"]
        self._use_reconstructed = config["use_reconstructed"]
        self._use_hyper_atten = config["use_hyper_atten"]

        # Embedding module
        embeddings = torch.nn.Embedding(self._voc_size, self._emb_size)
        embeddings.weight.data = truncated_normal(embeddings.weight.data, std=0.02)
        self.embedding = embeddings
        self.embeds_layer_normal = torch.nn.LayerNorm(normalized_shape=self._emb_size, eps=1e-12, elementwise_affine=True)
        self.embeds_dropout = torch.nn.Dropout(0.1)
        self.positional_encoding = PositionalEncoding(d_model=self._emb_size)

        # encoder module
        self.encoder = torch.nn.ModuleList()
        self.rec_MLP = torch.nn.ModuleList()
        self.p_dim = int(self._emb_size / self._num_view)
        self.c_dim = self._emb_size - self.p_dim
        for i in range(self._num_view):
            self.encoder.append(MVEncoder(
                self._emb_size,
                self._use_dynamic,
                self._batch_size,
                self._device,
                self._use_hyper_atten
            ))
            if self._use_reconstructed and self._use_disentangled:
                self.rec_MLP.append(MLP((self.p_dim + self.c_dim), self._emb_size, [self._emb_size],
                                   dropout_rate=0.1, activate_fun_type='GELU', use_layer_normal=True))
            elif self._use_reconstructed and not self._use_disentangled:
                self.rec_MLP.append(MLP(self._emb_size, self._emb_size, [self._emb_size],
                                   dropout_rate=0.1, activate_fun_type='GELU', use_layer_normal=True))

        # disentangled module
        if self._use_disentangled:
            self.common_MLP = torch.nn.ModuleList()
            self.private_MLP = torch.nn.ModuleList()
            self.mea_func = torch.nn.ModuleList()
            for i in range(self._num_view):
                self.common_MLP.append(MLP(self._emb_size, self.c_dim, [self._emb_size],
                                           dropout_rate=0.1, activate_fun_type='GELU', use_layer_normal=True))
                self.private_MLP.append(MLP(self._emb_size, self.p_dim, [self._emb_size],
                                            dropout_rate=0.1, activate_fun_type='GELU', use_layer_normal=True))
                self.mea_func.append(Measure_F(self.c_dim, self.p_dim,
                                               [self._emb_size] * self._phi_num_layers,
                                               [self._emb_size] * self._phi_num_layers))
            # self.p_fusion_MLP = MLP(self.p_dim * self._num_view, self._emb_size, [self._emb_size],
            #                    dropout_rate=0.1, activate_fun_type='GELU', use_layer_normal=True)
            self.concat_MLP = MLP((self.p_dim * self._num_view + self.c_dim), self._emb_size, [self._emb_size],
                                  dropout_rate=0.0, activate_fun_type='GELU', use_layer_normal=True)
        else:

            self.concat_MLP = MLP(self._emb_size * self._num_view, self._emb_size, [self._emb_size],
                                  dropout_rate=0.0, activate_fun_type='GELU', use_layer_normal=True)
            # self.concat_MLP = torch.nn.Linear(self._emb_size, self._emb_size)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=self._emb_size, eps=1e-7, elementwise_affine=True)

        # Link prediction module
        self.fc1 = torch.nn.Linear(self._emb_size, self._emb_size)
        self.fc1.weight.data = truncated_normal(self.fc1.weight.data, std=0.02)
        torch.nn.init.constant_(self.fc1.bias, 0.0)
        self.layer_norm2 = torch.nn.LayerNorm(normalized_shape=self._emb_size, eps=1e-7, elementwise_affine=True)
        self.fc2_bias = torch.nn.init.constant_(torch.nn.parameter.Parameter(torch.Tensor(self._voc_size)), 0.0)

        self.compute_loss = ComputeLoss(
            batch_size=self._batch_size,
            device=self._device,
            n_relation=self._n_relation,
            voc_size=self._voc_size,
            e_soft_label=self._e_soft_label,
            r_soft_label=self._r_soft_label
        )

    def forward(self, data, is_train=True):
        (orig_input_ids, orig_input_mask, mask_position, mask_label, mask_type, orig_input_type,
         nei_input_ids, nei_input_mask, nei_incidence_matrix) = data

        self._device = orig_input_ids.device
        batch_size = orig_input_ids.size(0)
        max_seq_len = orig_input_ids.size(1)
        mask_pos = mask_position.unsqueeze(1)

        # Encoder module
        enc_outputs, res, inp_embeds = self.encoder_module(nei_input_ids, nei_incidence_matrix, batch_size)
        rec_loss = res

        # Disentangled module
        if self._use_disentangled:
            common = []
            private = []
            for i in range(self._num_view):
                enc_output = enc_outputs[i]
                # hf_input_mask = orig_input_mask.unsqueeze(2).expand(-1, -1, self._emb_size)
                # hf_emb = enc_output[:, :max_seq_len, :].mul(hf_input_mask)
                hf_emb = enc_output[:, :max_seq_len, :]
                c = self.common_MLP[i](hf_emb)
                p = self.private_MLP[i](hf_emb)
                common.append(c)
                private.append(p)
            S = self.bulid_S(common)
            if self._use_reconstructed:
                for i in range(self._num_view):
                    # Compute the reconstructed loss
                    rec_node_embedding = self.rec_MLP[i](torch.cat([common[i], private[i]], dim=-1))
                    org_inp_embeds = inp_embeds[i][:, :max_seq_len, :]
                    rec_loss += self.reconstructed_loss(org_inp_embeds, rec_node_embedding)
            # # Compute the disentangled loss
            match_loss, corr_loss = self.disentangled_loss(common, private, S)

            # Fusion
            mask_p_pos = mask_pos[:, :, None].expand(-1, -1, self.p_dim)
            masked_p_vector = []
            for i in range(self._num_view):
                masked_p_vector.append(
                    torch.gather(input=private[i], dim=1, index=mask_p_pos).reshape([-1, private[i].size(-1)]))
            masked_p_vector = torch.cat(masked_p_vector, dim=-1)
            masked_vector = []
            mask_c_pos = mask_pos[:, :, None].expand(-1, -1, self.c_dim)
            masked_s_vector = torch.gather(input=S, dim=1, index=mask_c_pos).reshape([-1, S.size(-1)])
            masked_vector.append(masked_s_vector)
            masked_vector.append(masked_p_vector)
            # masked_vector.append(self.p_fusion_MLP(masked_p_vector))
            mask_token_embedding = self.concat_MLP(torch.cat(masked_vector, dim=-1))
            # mask_token_embedding = masked_s_vector
        else:
            mask_pos = mask_pos[:, :, None].expand(-1, -1, self._emb_size)
            masked_vectors = []
            for i in range(self._num_view):
                enc_output = enc_outputs[i]
                if self._use_reconstructed:
                    rec_node_embedding = self.rec_MLP[i](enc_output)
                    rec_loss += self.reconstructed_loss(inp_embeds[i], rec_node_embedding)
                masked_vectors.append(
                    torch.gather(input=enc_output, dim=1, index=mask_pos).reshape([-1, enc_output.size(-1)]))
            mask_token_embedding = self.concat_MLP(torch.cat(masked_vectors, dim=-1))

        # Link prediction module
        fc_out = self.prediction_layer(mask_token_embedding)
        # Compute the link prediction loss
        lp_loss, fc_out = self.compute_loss(mask_type, fc_out, mask_label)

        loss = lp_loss
        if self._use_disentangled and self._loss_mat:
            loss += self._loss_lamda * match_loss
        if self._use_disentangled and self._loss_cor:
            loss += corr_loss
        if self._use_reconstructed:
            loss += rec_loss

        return loss, fc_out

    def encoder_module(self, nei_input_ids, nei_incidence_matrix, batch_size):
        # Encoder module
        inp_embeds = []
        enc_outputs = []
        res = 0
        for i in range(self._num_view):
            view_id = int(self._view_id[i])  # Extracts the selected view
            input_ids = nei_input_ids[:, view_id, :]
            incidence_matrix = nei_incidence_matrix[:, view_id, :, :]

            input_embeds = self.embedding(input_ids)
            input_embeds = self.embeds_dropout(self.embeds_layer_normal(input_embeds))
            input_embeds = self.positional_encoding(input_embeds)

            if self._use_dynamic:
                enc_output, e = self.encoder[i](input_embeds, incidence_matrix)
                res += self.hypergraph_decoder(enc_output, e, incidence_matrix)
            else:
                # Building a Block diagonal matrix, for batch training.
                x = input_embeds.view(batch_size * input_embeds.size(1), input_embeds.size(2))
                hyperedge_index = self.batch_to_block_diagonal_sparse(incidence_matrix)
                enc_output, _ = self.encoder[i](x, hyperedge_index)
                enc_output = enc_output.view(batch_size, input_embeds.size(1), input_embeds.size(2))
            enc_outputs.append(enc_output)
            inp_embeds.append(input_embeds)

        return enc_outputs, res, inp_embeds

    def hypergraph_decoder(self, _enc_out, e, incidence_matrix_T):
        hye_enc_out = e
        sigmoid = torch.nn.Sigmoid()
        re_incidence_matrix = sigmoid(_enc_out.bmm(hye_enc_out.transpose(-2, -1)))
        mseloss = torch.nn.MSELoss()
        re_err = mseloss(re_incidence_matrix, incidence_matrix_T.float())
        return re_err

    def prediction_layer(self, mask_token_embedding):
        # transform: fc1
        h_masked = self.fc1(mask_token_embedding)
        h_masked = torch.nn.GELU()(h_masked)
        # transform: layer norm
        h_masked = self.layer_norm2(h_masked)
        # transform: fc2 weight sharing
        fc_out = torch.nn.functional.linear(h_masked, self.embedding.weight, self.fc2_bias)
        return fc_out

    def reconstructed_loss(self, ori, rep):
        l = torch.nn.MSELoss()
        return l(rep, ori)

    def disentangled_loss(self, c, p, s):
        l = torch.nn.MSELoss(reduction='sum')
        match_loss = 0
        for i in range(self._num_view):
            match_loss += l(c[i], s) / c[i].shape[1]

        # Independence regularizer loss
        phi_c_list = []
        psi_p_list = []
        for i in range(self._num_view):
            phi_c, psi_p = self.mea_func[i](c[i], p[i])
            phi_c_list.append(phi_c)
            psi_p_list.append(psi_p)
        # Correlation
        corr_loss = 0
        for i in range(len(phi_c_list)):
            corr_loss += self.compute_corr(phi_c_list[i], psi_p_list[i])
        assert torch.isnan(corr_loss).sum() == 0, print(corr_loss)
        return match_loss, corr_loss

    def compute_corr(self, x1, x2):
        assert torch.isnan(x2).sum() == 0, print(x2)
        # Subtract the mean
        x1_mean = torch.mean(x1, 0, True)
        x1 = x1 - x1_mean
        x2_mean = torch.mean(x2, 0, True)
        x2 = x2 - x2_mean

        # Compute the cross correlation
        sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
        sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
        corr = torch.abs(torch.mean(x1 * x2)) / ((sigma1 * sigma2) + 1e-8)
        assert torch.isnan(corr).sum() == 0, print(corr)
        return corr

    def bulid_S(self, common):
        # Buliding common S
        FF = []
        FF.append(torch.cat(common, dim=-1))
        FF = torch.cat(FF, dim=0)
        FF = FF - torch.mean(FF, 1, True)
        h = []
        for i in range(self._num_view):
            h.append(FF[:, :, i * self.c_dim:(i + 1) * self.c_dim])
        FF = torch.stack(h, dim=3)

        a = torch.sum(FF, dim=-1)
        # The SVD step
        U, _, T = torch.svd(a)
        S = torch.matmul(U, T.transpose(-2, -1))
        S = S * (FF.shape[1]) ** 0.5
        return S

    def trans_sparse_coo_tensor(self, sparse_batch):
        batch_size, N, M = sparse_batch.shape
        device = sparse_batch.device

        indices = sparse_batch._indices()
        values = sparse_batch._values()

        new_indices = []
        new_values = []
        offset = 0

        for i in range(batch_size):
            batch_indices = indices[:, (indices[0] == i).nonzero(as_tuple=True)[0]]
            batch_indices = batch_indices[1:]
            batch_indices += offset

            new_indices.append(batch_indices)
            new_values.append(values[(indices[0] == i).nonzero(as_tuple=True)[0]])

            offset += N

        new_indices = torch.cat(new_indices, dim=1)
        new_values = torch.cat(new_values)

        new_size = (batch_size * N, batch_size * M)

        new_sparse_tensor = torch.sparse_coo_tensor(new_indices, new_values, new_size).to(device)

        return new_sparse_tensor

    def batch_to_block_diagonal_sparse(self, tensor):
        """
        Convert a tensor of shape [batch_size, N, M] to a block diagonal matrix
        and then convert it to a torch.sparse_coo_tensor.

        Args:
        - tensor (torch.Tensor): Input tensor of shape [batch_size, N, M]

        Returns:
        - torch.sparse_coo_tensor: Output sparse tensor of shape [batch_size*N, batch_size*M]
        """
        batch_size, N, M = tensor.shape
        total_N = batch_size * N
        total_M = batch_size * M

        indices = []
        values = []

        for b in range(batch_size):
            batch_indices = torch.nonzero(tensor[b], as_tuple=False)
            batch_indices[:, 0] += b * N
            batch_indices[:, 1] += b * M
            indices.append(batch_indices)
            values.append(tensor[b][tensor[b] != 0])

        indices = torch.cat(indices, dim=0).t()
        values = torch.cat(values, dim=0)

        sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(total_N, total_M))
        return sparse_tensor




def truncated_normal(t, mean=0.0, std=0.01):
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
        cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
        if not torch.sum(cond):
            break
        t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
    return t
