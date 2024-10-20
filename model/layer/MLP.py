import torch.nn


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim_list,
                 activate_fun_type='ReLU', dropout_rate=0.0, use_layer_normal=False):
        super(MLP, self).__init__()
        self.middle_dim_list = middle_dim_list
        self.use_layer_normal = use_layer_normal
        self.net = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer_num = len(middle_dim_list) + 1
        if self.layer_num == 2:
            self.net.append(torch.nn.Linear(input_dim, middle_dim_list[0]))
            self.net.append(torch.nn.Linear(middle_dim_list[0], output_dim))
        elif self.layer_num > 2:
            self.net.append(torch.nn.Linear(input_dim, middle_dim_list[0]))
            for i in range(self.layer_num-2):
                self.net.append(torch.nn.Linear(middle_dim_list[i], middle_dim_list[i+1]))
            self.net.append(torch.nn.Linear(middle_dim_list[-1], output_dim))
        else:
            raise ValueError("Invalid `layer_num`.")

        if activate_fun_type == 'ReLU':
            self.activate_fun = torch.nn.ReLU(inplace=True)
        elif activate_fun_type == 'GELU':
            self.activate_fun = torch.nn.GELU()
        elif activate_fun_type == 'ELU':
            self.activate_fun = torch.nn.ELU(inplace=True)
        elif activate_fun_type == 'LeakyReLU':
            self.activate_fun = torch.nn.LeakyReLU(inplace=True)
        else:
            raise ValueError("Invalid `activate_fun_type`.")

        if self.use_layer_normal:
            self.layer_normal = torch.nn.ModuleList()
            for i in range(self.layer_num-1):
                self.layer_normal.append(torch.nn.LayerNorm(normalized_shape=self.middle_dim_list[i], eps=1e-7, elementwise_affine=True))

    def forward(self, x):
        for i in range(len(self.net)-1):
            x = self.activate_fun(self.net[i](x))
            if self.use_layer_normal:
                x = self.layer_normal[i](x)
            x = self.dropout(x)
        # For the last layer
        y = self.net[-1](x)
        return y

