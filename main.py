import argparse
import logging
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from model.HGNN import HGNNModel
from reader.entity.GNNDataset import GNNDataset
from reader.vocab_reader import Vocabulary
from utils.args import ArgumentGroup
from utils.evaluation import predict_for_evaluation
from utils.lr_scheduler import get_linear_schedule_with_warmup
from utils.train_args import set_default_args
from reader.data_reader import load_data, generate_ground_truth
from utils.sample import random_sample

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(logger.getEffectiveLevel())
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')

# =====================  Argument Parser  =============================== #
parser = argparse.ArgumentParser()
data_g = ArgumentGroup(parser, "data",     "Data paths, vocab paths and data processing options.")
data_g.add_arg("dataset_name",            str,    "jf17k",   "Dataset name")
data_g.add_arg("train_file",              str,    None,      "Data for training.")
data_g.add_arg("valid_file",              str,    None,      "Data for valid.")
data_g.add_arg("test_file",               str,    None,      "Data for test.")
data_g.add_arg("predict_file",            str,    None,      "Data for prediction.")
data_g.add_arg("ground_truth_path",       str,    None,      "Path to ground truth.")
data_g.add_arg("vocab_path",              str,    None,      "Path to vocabulary.")
data_g.add_arg("vocab_dict_path",         str,    None,      "Path of vocabulary(dict).")
data_g.add_arg("vocab_size",              int,    None,      "Size of vocabulary.")
data_g.add_arg("num_relations",           int,    None,      "Number of relations.")
data_g.add_arg("max_seq_len",             int,    None,      "Max sequence length.")
data_g.add_arg("max_arity",               int,    None,      "Max arity.")
data_g.add_arg("entity_soft_label",       float,  0.1,       "Label smoothing rate for masked entities.")
data_g.add_arg("relation_soft_label",     float,  0.1,       "Label smoothing rate for masked relations.")
data_g.add_arg("sample_rate",             float,  1,         "Train dataset sample rate")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",                     bool,   True,          "If set, use GPU for training.")
run_type_g.add_arg("device",                       str,    "1",           "{0123}^n,1<=n<=4,the first cuda is used as master device and others are used for data parallel")
run_type_g.add_arg("model_type",                   str,    "GNN",         "")
run_type_g.add_arg("model_name_or_path",           str,    "",    "")
run_type_g.add_arg("stop",                         bool,   True,          "")
run_type_g.add_arg("stop_step",                    int,    10,            "")
run_type_g.add_arg("iteration",                    int,    1,             "")
run_type_g.add_arg("start_step",                   int,    -1,             "")
run_type_g.add_arg("save_model",                   bool,   True,          "")
run_type_g.add_arg("use_processed_data",           bool,   True,          "Loading new data")
run_type_g.add_arg("use_checkpoints",              bool,   False,          "")
run_type_g.add_arg("checkpoints",                  str,    "",   "Path to save checkpoints.")
run_type_g.add_arg("checkpoints_file",             str,    "",   "")

model_g = ArgumentGroup(parser, "model",    "model and checkpoint configuration.")
model_g.add_arg("hidden_size",             int,    256,      "Hidden size.")
model_g.add_arg("sample_k",                int,    5,        "Hypergraph sample k.")
model_g.add_arg("use_dynamic",             bool,   True,     "True-use hypergraph transformer encoder; False-use HGCN encoder.")
model_g.add_arg("use_hyper_atten",         bool,   False,    "True-use Hypergraph Attention; False-use Hypergraph Convolution.")
model_g.add_arg("num_view",                int,    4,        "The number of view. ")
model_g.add_arg("view_id",                 str,    "0123",   "0-h, 1-r, 2-t, 3-av. ")
model_g.add_arg("use_disentangled",        bool,   True,     "Use disentangled representation learning ")
model_g.add_arg("use_reconstructed",       bool,   True,     "Use reconstructed learning ")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("batch_size",        int,    128,                   "Batch size.")
train_g.add_arg("epoch",             int,    100,                    "Number of training epochs.")
train_g.add_arg("learning_rate",     float,  3e-4,                   "Learning rate with warmup.")
train_g.add_arg("lr_scheduler",      str,    "linear_warmup_decay",  "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("warmup_proportion", float,  0.1,                    "Proportion of training steps for lr warmup.")
train_g.add_arg("weight_decay",      float,  0.01,                   "Weight decay rate for L2 regularizer.")
train_g.add_arg("loss_lamda",        float,  0.00001,                 "Loss function weight")
train_g.add_arg("loss_mat",          bool,  True,                    "use matching loss")
train_g.add_arg("loss_cor",          bool,  True,                    "use correlation loss")

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps",          int,    1000,    "Step intervals to print loss.")
log_g.add_arg("verbose",             bool,   False,   "Whether to output verbose log.")
log_g.add_arg("logs_save_dir",       str,    "./logs",      "Path to save logs.")



args = parser.parse_args()
# =====================  Argument Parser END  =============================== #


def train_for_GNN(args, config, device, devices):
    # ************* load dataset ***************
    vocabulary = Vocabulary(args.vocab_path, args.num_relations, args.vocab_size - args.num_relations - 2)
    all_facts = generate_ground_truth(args.ground_truth_path, vocabulary, args.max_seq_len)

    logger.info("loading train dataloader")
    processed_data_path = './data/{}/GNN_processed/dataloader.pth'.format(args.dataset_name)
    if args.use_processed_data:
        local_data = torch.load(processed_data_path, map_location=device)
        local_train_data = local_data['train_data']
        local_test_data = local_data['test_data']
        local_train_data = random_sample(local_train_data, args.sample_rate)
        train_data = GNNDataset(local_train_data)
        test_data = GNNDataset(local_test_data)
    else:
        if not os.path.exists('./data/{}/GNN_processed'.format(args.dataset_name)):
            os.makedirs('./data/{}/GNN_processed'.format(args.dataset_name))
        train_data = load_data(args, args.train_file, vocabulary, None, device)
        test_data = load_data(args, args.test_file, vocabulary, None, device)
        torch.save({
            'train_data': train_data.data,
            'test_data': test_data.data
        }, processed_data_path)

    # if args.dataset_name == "wikipeople":
    #     train_pyreader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
    #                                 batch_sampler=partial(RandomSampler, num_samples=int(len(train_data) / 10)))
    # else:
    train_pyreader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_pyreader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    if len(devices) > 1:
        model = torch.nn.DataParallel(HGNNModel(config), device_ids=devices)
        model.to(device)
    else:
        model = HGNNModel(config, device).to(device)
    logger.info(model)

    return train_pyreader, test_pyreader, all_facts, model


def trainer(args, time_):
    """

    :param args:
    :param time_:
    :return:
    """
    args = set_default_args(args, args.dataset_name)
    config = vars(args)
    devices = []
    if args.use_cuda:
        # device = torch.device(args.device)
        # torch.cuda.set_device(0)
        # prepare GPU or GPUs
        device = torch.device(f"cuda:{args.device[0]}")
        for i in range(len(args.device)):
            devices.append(torch.device(f"cuda:{args.device[i]}"))
    else:
        device = torch.device("cpu")
        config["device"] = "cpu"
    # args display
    for k, v in vars(args).items():
        logger.info(k + ':' + str(v))

    if args.model_type in ["GNN"]:
        train_pyreader, test_pyreader, all_facts, model = train_for_GNN(args, config, device, devices)
    else:
        raise ValueError("Invalid `model_type`.")

    t_total = len(train_pyreader) * args.epoch
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)

    iteration_start = 1
    if args.use_checkpoints:
        logger.info("loal checkpoint model for train")
        checkpoint = torch.load(os.path.join(args.checkpoints, "{}".format(args.checkpoints_file)))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iteration_start = checkpoint['epoch'] + 1

    # ************* Training ********************
    max_entity_mmr = 0
    max_updata_num = args.stop_step
    for iteration in range(iteration_start, args.epoch):
        logger.info("iteration " + str(iteration))
        t1_strat = time.time()
        # --------- train -----------
        for j, data in tqdm(enumerate(train_pyreader), total=len(train_pyreader)):
            model.train()
            optimizer.zero_grad()
            loss, fc_out = model(data)

            if len(devices) > 1:
                loss = torch.sum(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if j % 100 == 0:
                logger.info(str(j) + ' , loss: ' + str(loss.item()))
        # --------- validation -----------
        if iteration % args.iteration == 0 and iteration > args.start_step:
            logger.info("Train time = {:.3f} s".format(round(time.time() - t1_strat, 2)))
            # Start validation and testing
            logger.info("Start validation")
            model.eval()
            with torch.no_grad():
                entity_mmr = predict_for_evaluation(model, test_pyreader, all_facts, device, logger,
                                                    args.model_type)

            t1_end = time.time()
            t1 = round(t1_end - t1_strat, 2)
            logger.info("Iteration time = {:.3f} s".format(t1))
            # save model
            if entity_mmr > max_entity_mmr:
                max_entity_mmr = entity_mmr
                max_updata_num = args.stop_step
                logger.info("=============== Best performance ============== ")
                if args.save_model:
                    torch.save({
                        'epoch': iteration,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(args.checkpoints, "{}_{}_model_{}.tar".format(args.dataset_name, args.model_type, time_)))
                    logger.info("save model to {}".format(
                        os.path.join(args.checkpoints, "{}_{}_model_{}.tar".format(args.dataset_name, args.model_type, time_))))
            else:
                max_updata_num = max_updata_num - 1
                if max_updata_num == 0 and args.stop:
                    logger.info("The model has converged.")
                    break




if __name__ == '__main__':
    # log file
    data_file_path = str(time.strftime("%Y-%m-%d", time.localtime()))
    logs_save_dir = os.path.join(args.logs_save_dir, args.dataset_name, data_file_path)
    if not os.path.exists(logs_save_dir):
        os.makedirs(logs_save_dir)
    time_ = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    fileHandler = logging.FileHandler(
        os.path.join(logs_save_dir, '{}_{}_{}.log'.format(time_, args.dataset_name, args.device)))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    trainer(args, time_)

