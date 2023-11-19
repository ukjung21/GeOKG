
import argparse
import json
import logging
import os

import torch
import torch.optim
import torch.nn

import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
from utils.train import get_savedir, avg_both, format_metrics, count_params
import pickle
import numpy as np
import csv

parser = argparse.ArgumentParser(
    description="Knowledge Graph Embedding"
)
parser.add_argument(
    "--dataset", default="WN18RR",
    help="Knowledge Graph dataset"
)
parser.add_argument(
    "--model", default="GIE", help="Knowledge Graph embedding model"
)
parser.add_argument(
    "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
)
parser.add_argument(
    "--reg", default=0, type=float, help="Regularization weight"
)
parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adagrad",
    help="Optimizer"
)
parser.add_argument(
    "--max_epochs", default=50, type=int, help="Maximum number of epochs to train for"
)
parser.add_argument(
    "--patience", default=10, type=int, help="Number of epochs before early stopping"
)
parser.add_argument(
    "--valid", default=3, type=float, help="Number of epochs before validation"
)
parser.add_argument(
    "--rank", default=1000, type=int, help="Embedding dimension"
)
parser.add_argument(
    "--batch_size", default=1000, type=int, help="Batch size"
)
parser.add_argument(
    "--neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate"
)
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)
parser.add_argument(
    "--learning_rate", default=1e-1, type=float, help="Learning rate"
)
parser.add_argument(
    "--gamma", default=0, type=float, help="Margin for distance-based losses"
)
parser.add_argument(
    "--bias", default="constant", type=str, choices=["constant", "learn", "none"], help="Bias type (none for no bias)"
)
parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision"
)
# parser.add_argument('--margin', type=float, default=1.5)
# parser.add_argument('--u1', type=float, default=0.05)
# parser.add_argument('--u2', type=float, default=10.0)
# parser.add_argument('--lam', type=float, default=3.0)
parser.add_argument('--check', type=str, default=None)
parser.add_argument('--prtd', type=str, default=None)

parser.add_argument(
    "--double_neg", action="store_true",
    help="Whether to negative sample both head and tail entities"
)
parser.add_argument(
    "--debug", action="store_true",
    help="Only use 1000 examples for debugging"
)
parser.add_argument(
    "--multi_c", action="store_true", help="Multiple curvatures per relation"
)

def train(args):
    save_dir = get_savedir(args.model, args.dataset)

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.log")
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))

    dataset_path = os.path.join(r'data/', args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()


    logging.info("\t " + str(dataset.get_shape()))
    train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson)

    model = getattr(models, args.model)(args)
    model = torch.nn.DataParallel(model)
    
    if args.check:
        model.load_state_dict(torch.load((args.check), map_location='cuda'))
        model.cuda()
        model.eval()
        
        with open(file='data/'+args.dataset+'/test_neg.pickle', mode='rb') as f:
            neg_trp = pickle.load(f)
        
        # test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
        test_metrics = model.compute_metrics(test_examples, filters)
        test_metrics = {'MR': test_metrics[0]['rhs'], 'MRR': test_metrics[1]['rhs'], 'hits@[1,10,50,100]': test_metrics[2]['rhs']}
        logging.info(format_metrics(test_metrics, split="test"))
        
        entities = model.entity
        relations = model.rel
        np.save(os.path.join(save_dir, 'entity_embedding.npy'), entities.weight.detach().cpu().numpy())
        np.save(os.path.join(save_dir, 'relation_embedding.npy'), relations.weight.detach().cpu().numpy())

        # roc, pr, max_f1 = model.compute_roc(test_examples, neg_trp, save_path=save_dir, num_rel=args.sizes[1])
        roc, pr, max_f1 = model.compute_roc(test_examples, neg_trp, save_path=save_dir, batch_size=args.batch_size, num_rel=args.sizes[1]//2)
        # mic_f1, wa_f1, accuracy = model.compute_rel_acc(test_examples, save_path=save_dir, num_rel=args.sizes[1])
        mic_f1, wa_f1, accuracy = model.compute_rel_acc(test_examples, save_path=save_dir, batch_size=args.batch_size, num_rel=args.sizes[1]//2)
        
        eval_list = [args.dataset, args.model, round(roc,4), round(pr,4), round(max_f1,4), round(mic_f1,4), round(wa_f1,4)]
        for _, j in accuracy.items():
            eval_list.append(round(j,4))
        with open('LOG_DIR/eval_metrics.tsv', mode='a', newline='') as f:
            wr = csv.writer(f, delimiter='\t')
            wr.writerow(eval_list)
    
    else:
        total = count_params(model)
        logging.info("Total number of parameters {}".format(total))
        device = "cuda"
        model.to(device)

        regularizer = getattr(regularizers, args.regularizer)(args.reg)
        optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
        optimizer = KGOptimizer(args, model, regularizer, optim_method, args.batch_size, args.neg_sample_size,
                                bool(args.double_neg))
        counter = 0
        best_mrr = None
        best_epoch = None
        logging.info("\t Start training")
        for step in range(args.max_epochs):

            model.train()
            train_loss = optimizer.epoch(train_examples)
            logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

            model.eval()
            valid_loss = optimizer.calculate_valid_loss(valid_examples)
            logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

            if (step + 1) % args.valid == 0:
                # valid_metrics = avg_both(*model.module.compute_metrics(valid_examples, filters))
                valid_metrics = model.module.compute_metrics(valid_examples, filters, batch_size=args.batch_size)
                valid_metrics = {'MR': valid_metrics[0]['rhs'], 'MRR': valid_metrics[1]['rhs'], 'hits@[1,10,50,100]': valid_metrics[2]['rhs']}
                logging.info(format_metrics(valid_metrics, split="valid"))

                valid_mrr = valid_metrics["MRR"]
                if not best_mrr or valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    counter = 0
                    best_epoch = step
                    logging.info("\t Saving model at epoch {} in {}".format(step, save_dir))
                    torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
                    model.cuda()
                else:
                    counter += 1
                    if counter == args.patience:
                        logging.info("\t Early stopping")
                        break
                    elif counter == args.patience // 2:
                        pass


        logging.info("\t Optimization finished")
        if not best_mrr:
            torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
        else:
            logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
            model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
        model.cuda()
        model.eval()

        logging.info(format_metrics(valid_metrics, split="valid"))

        test_metrics = model.module.compute_metrics(test_examples, filters, batch_size=args.batch_size)
        test_metrics = {'MR': test_metrics[0]['rhs'], 'MRR': test_metrics[1]['rhs'], 'hits@[1,10,50,100]': test_metrics[2]['rhs']}
        logging.info(format_metrics(test_metrics, split="test"))
            
        entities = model.module.entity
        relations = model.module.rel
        head_bias = model.module.bh
        tail_bias = model.module.bt
        
        np.save(os.path.join(save_dir, 'entity_embedding.npy'), entities.weight.detach().cpu().numpy())
        np.save(os.path.join(save_dir, 'relation_embedding.npy'), relations.weight.detach().cpu().numpy())
        np.save(os.path.join(save_dir, 'bh_embedding.npy'), head_bias.weight.detach().cpu().numpy())
        np.save(os.path.join(save_dir, 'bt_embedding.npy'), tail_bias.weight.detach().cpu().numpy())
        diag_relations = model.module.rel_diag
        np.save(os.path.join(save_dir, 'diag_relation_embedding.npy'), diag_relations.weight.detach().cpu().numpy())
        
        if not args.model.endswith('E') or args.model=='AttE':
            
            diag1_relations = model.module.rel_diag1
            diag2_relations = model.module.rel_diag2
            
            np.save(os.path.join(save_dir, 'diag1_relation_embedding.npy'), diag1_relations.weight.detach().cpu().numpy())
            np.save(os.path.join(save_dir, 'diag2_relation_embedding.npy'), diag2_relations.weight.detach().cpu().numpy())
        
        # roc, pr, max_f1 = model.module.compute_roc(test_examples, neg_trp, save_path=save_dir, num_rel=args.sizes[1])
        hits = test_metrics['hits@[1,10,50,100]'].numpy()
        eval_list = [args.dataset, args.model, '', '{:.3f}'.format(test_metrics['MRR']), \
            '{:.3f}'.format(hits[0]), '{:.3f}'.format(hits[1]), '{:.3f}'.format(hits[2]), '{:.3f}'.format(hits[3])]
        
        if (args.dataset).startswith('GO'):
            with open(file='data/'+args.dataset+'/test_neg.pickle', mode='rb') as f:
                neg_trp = pickle.load(f) 
            roc, pr, max_f1 = model.module.compute_roc(test_examples, neg_trp, save_path=save_dir, batch_size=args.batch_size, num_rel=args.sizes[1]//2)        
            eval_list.append(['{:.3f}'.format(roc), '{:.3f}'.format(pr), '{:.3f}'.format(max_f1)])
        
            if (not args.dataset.endswith('1018')):
                # mic_f1, wa_f1, accuracy = model.module.compute_rel_acc(test_examples, save_path=save_dir, num_rel=args.sizes[1])
                mic_f1, wa_f1, accuracy = model.module.compute_rel_acc(test_examples, save_path=save_dir, batch_size=args.batch_size, num_rel=args.sizes[1]//2)
                eval_list.append(['{:.3f}'.format(mic_f1), '{:.3f}'.format(wa_f1)])
                for _, j in accuracy.items():
                    eval_list.append('{:.3f}'.format(j))

        with open('LOG_DIR/eval_metrics.tsv', mode='a', newline='') as f:
            wr = csv.writer(f, delimiter='\t')
            wr.writerow(eval_list)


if __name__ == "__main__":
    train(parser.parse_args())
