class HFactFeatureGNN(object):
    """
    A single set of features used for training/test.
    """

    def __init__(self,
                 feature_id,
                 example_id,
                 input_ids,
                 input_mask,
                 input_label,
                 mask_position,
                 mask_label,
                 mask_type,
                 h_input_ids,
                 h_input_mask,
                 h_incidence_matrix,
                 arity,
                 ):
        """
        Construct NaryFeature.

        Args:
            feature_id: unique feature id
            example_id: corresponding example id
            input_sentence: input sequence of sentence
            input_mask: input sequence mask
            mask_position: position of masked token
            mask_label: label of masked token
            mask_type: type of masked token,
                1 for entities (values) and -1 for relations (attributes)
            arity: arity of the corresponding example
        """
        self.feature_id = feature_id
        self.example_id = example_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_label = input_label
        self.mask_position = mask_position
        self.mask_label = mask_label
        self.mask_type = mask_type
        self.h_input_ids = h_input_ids
        self.h_input_mask = h_input_mask
        self.h_incidence_matrix = h_incidence_matrix
        self.arity = arity
