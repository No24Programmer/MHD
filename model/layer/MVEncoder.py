import torch.nn
from torch_geometric.nn import HypergraphConv

from model.layer.transformer import transformer_encoder


class MVEncoder(torch.nn.Module):
    def __init__(self, emb_size, use_dynamic, batch_size, device, use_hyper_atten=False):
        super(MVEncoder, self).__init__()
        self._emb_size = emb_size
        self._use_dynamic = use_dynamic
        self._batch_size = batch_size
        self._device = device
        self._use_hyper_atten = use_hyper_atten

        self.edges_mu = torch.nn.Parameter(torch.randn(1, self._emb_size))
        self.edges_logsigma = torch.nn.Parameter(torch.zeros(1, self._emb_size))
        torch.nn.init.xavier_uniform_(self.edges_logsigma)

        if self._use_dynamic:
            # positional encoding
            self.transformer_encoder = transformer_encoder(
                n_layer=2,
                n_head=4,
                d_model=self._emb_size,
                d_inner_hid=self._emb_size*2,
                prepostprocess_dropout=0.1,
                attention_dropout=0.1
            )
        else:
            if use_hyper_atten:
                self.hyperedge_weight = torch.nn.Parameter(torch.normal(0, 1, size=()))
                self.HGCN = HypergraphConv(
                    in_channels=self._emb_size,
                    out_channels=self._emb_size,
                    use_attention=True,
                )
            else:
                self.HGCN = HypergraphConv(
                    in_channels=self._emb_size,
                    out_channels=self._emb_size,
                )

    def forward(self, input_embeds, incidence_matrix):
        if self._use_dynamic:
            hyperedge_emb = self.hyperedge_init(incidence_matrix, self._batch_size)
            enc_output, e = self.transformer_encoder(
                enc_input=input_embeds,
                adj_matrix=incidence_matrix.to_dense(),
                e=hyperedge_emb,
            )
        else:
            if self._use_hyper_atten:
                hyperedge_emb = self.hyperedge_init(incidence_matrix, self._batch_size)
                enc_output = self.HGCN(
                    x=input_embeds,
                    hyperedge_index=incidence_matrix._indices(),
                    hyperedge_attr=hyperedge_emb,
                )
                e = 0
            else:
                enc_output = self.HGCN(
                    x=input_embeds,
                    hyperedge_index=incidence_matrix._indices(),
                )
                e = 0
        return enc_output, e

    def hyperedge_init(self, incidence_matrix, batch_size):
        # Use dynamic hypergraph
        # if self._use_dynamic:
        mu = self.edges_mu.expand(incidence_matrix.size()[-1], -1)
        sigma = self.edges_logsigma.exp().expand(incidence_matrix.size()[-1], -1)
        e = mu + sigma * torch.randn((batch_size, mu.size(0), mu.size(1)), device=self._device)
        # else:
        #     e = None
        return e
