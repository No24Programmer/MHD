

def get_max_node(dataset, k=10):
    if k == 1:
        max_node_dict = {
            "jf17k": 35,
            "wikipeople": 30,
            "wd50k": 35,
        }
    elif k == 5:
        max_node_dict = {
            "jf17k": 55,
            "wikipeople": 70,
            "wd50k": 120,
            "wikipeople-": 45,
        }
    elif k == 10:
        # jf17k: k = 10, max_node_num = 55
        # wikipeople: k = 10, max_node_num = 65
        max_node_dict = {
            "jf17k": 70,
            "wikipeople-": 45,
            "wd50k": 120,
        }
    elif k == 15:
        max_node_dict = {
            "jf17k": 65,
            "wikipeople": 90,
            "wd50k": 130,
        }
    elif k == 20:
        max_node_dict = {
            "jf17k": 90,
            "wikipeople": 100,
            "wd50k": 135,
        }

    max_node = 0
    if dataset in max_node_dict.keys():
        max_node = max_node_dict[dataset]
    return max_node


def set_default_args(args, dataset_name):
    # Set data paths, vocab paths and data processing options.
    args.train_file = "./data/{}/train.json".format(dataset_name)
    # args.valid_file = "./data/{}/valid.json".format(dataset_name)
    # args.train_file = "./data/{}/train+valid.json".format(dataset_name)
    args.valid_file = "./data/{}/test.json".format(dataset_name)
    args.test_file = "./data/{}/test.json".format(dataset_name)

    args.predict_file = "./data/{}/test.json".format(dataset_name)
    args.ground_truth_path = "./data/{}/all.json".format(dataset_name)
    args.vocab_path = "./data/{}/vocab.txt".format(dataset_name)
    # args.vocab_path = "./data/{}/vocab_str.txt".format(dataset_name)
    args.vocab_dict_path = "./data/{}/vocab_str.json".format(dataset_name)


    if dataset_name == "wikipeople":
        args.batch_size = 256
        args.vocab_size = 47960
        args.num_relations = 193
        args.max_seq_len = 15   # 17
        args.max_arity = 9
        args.epoch = 200
        args.loss_lamda = 0.00001
        args.sample_rate = 1

    elif dataset_name == "wikipeople-":
        args.batch_size = 512
        args.vocab_size = 35005  # 34205
        args.num_relations = 178
        args.max_seq_len = 13
        args.max_arity = 7


    elif dataset_name == "jf17k":
        args.batch_size = 512
        # args.vocab_size = 28526  # 28969  #  29148
        args.vocab_size = 29148  # 28526  #  29148
        args.num_relations = 501  # 322 # 501
        args.max_seq_len = 11  # 7
        args.max_arity = 6
        args.epoch = 100
        args.loss_lamda = 1e-6
        args.hidden_size = 256
        args.learning_rate = 1e-3

    elif dataset_name == "wd50k":
        args.batch_size = 128
        args.vocab_size = 47688  # 45802
        args.num_relations = 531
        args.max_seq_len = 18  # 63
        args.max_arity = 32
        args.loss_lamda = 0.0001

    return args

