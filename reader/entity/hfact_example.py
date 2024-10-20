

class HFactExample(object):
    """
    A single training/test example of hyper-relational fact.
    """

    def __init__(self,
                 arity,
                 node_num,
                 relation,
                 head,
                 tail,
                 auxiliary_info=None):
        """
        Construct NaryExample.

        Args:
            arity (mandatory): arity of a given fact
            node_num (mandatory): node number of a given fact
            relation (mandatory): primary relation
            head (mandatory): primary head entity (subject)
            tail (mandatory): primary tail entity (object)
            auxiliary_info (optional): auxiliary attribute-value pairs,
                with attributes and values sorted in alphabetical order
        """
        self.arity = arity
        self.node_num = node_num
        self.relation = relation
        self.head = head
        self.tail = tail
        self.auxiliary_info = auxiliary_info
