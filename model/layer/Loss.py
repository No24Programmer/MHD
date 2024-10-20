import torch


class ComputeLoss(torch.nn.Module):
    def __init__(self, batch_size, device, n_relation, voc_size, e_soft_label, r_soft_label):
        super(ComputeLoss, self).__init__()
        self.batch_size = batch_size
        self._device = device
        self._n_relation = n_relation
        self._voc_size = voc_size
        self._e_soft_label = e_soft_label
        self._r_soft_label = r_soft_label
        self.link_prediction_loss = SoftmaxWithCrossEntropy()

    def forward(self, mask_type, fc_out, mask_label):
        # type_indicator [vocab_size,(yes1 or no0)]
        special_indicator = torch.empty(self.batch_size, 2).to(self._device)
        torch.nn.init.constant_(special_indicator, -1)
        relation_indicator = torch.empty(self.batch_size, self._n_relation).to(self._device)
        torch.nn.init.constant_(relation_indicator, -1)
        entity_indicator = torch.empty(self.batch_size, (self._voc_size - self._n_relation - 2)).to(self._device)
        torch.nn.init.constant_(entity_indicator, 1)
        type_indicator = torch.cat((relation_indicator, entity_indicator), dim=1).to(self._device)
        mask_type = mask_type.unsqueeze(1)
        type_indicator = torch.mul(type_indicator, mask_type)
        type_indicator = torch.cat([special_indicator, type_indicator], dim=1)
        type_indicator = torch.nn.functional.relu(type_indicator)

        fc_out_mask = 1000000.0 * (type_indicator - 1.0)
        fc_out = torch.add(fc_out, fc_out_mask)

        one_hot_labels = torch.nn.functional.one_hot(mask_label, self._voc_size)
        type_indicator = torch.sub(type_indicator, one_hot_labels)
        num_candidates = torch.sum(type_indicator, dim=1)

        soft_labels = ((1 + mask_type) * self._e_soft_label +
                       (1 - mask_type) * self._r_soft_label) / 2.0
        soft_labels = soft_labels.expand(-1, self._voc_size)
        soft_labels = soft_labels * one_hot_labels + (1.0 - soft_labels) * \
                      torch.mul(type_indicator, 1.0 / torch.unsqueeze(num_candidates, 1))

        link_prediction_loss = self.link_prediction_loss(logits=fc_out, label=soft_labels)

        return link_prediction_loss, fc_out


class SoftmaxWithCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(SoftmaxWithCrossEntropy, self).__init__()

    def forward(self, logits, label):
        logprobs = torch.nn.functional.log_softmax(logits, dim=1)
        loss = -1.0 * torch.sum(torch.mul(label, logprobs), dim=1).squeeze()
        loss = torch.mean(loss)
        return loss
