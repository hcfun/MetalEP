from torch.utils.tensorboard import SummaryWriter
from utils import Log
import json
from data import EvalDataset, ValidData, TestData
import pickle
import os
from kge_model import KGEModel
from collections import defaultdict as ddict
import torch
from torch.utils.data import DataLoader
import csv


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # writer and logger
        self.name = args.exp_name
        self.writer = SummaryWriter(os.path.join(args.tb_log_dir, self.name))
        self.logger = Log(args.log_dir, self.name).get_logger()
        self.logger.info(json.dumps(vars(args)))

        # state dir
        self.state_path = os.path.join(args.state_dir, self.name)
        if not os.path.exists(self.state_path):
            os.makedirs(self.state_path)

        # load data
        self.data = pickle.load(open(args.data_path, 'rb'))
        args.num_ent = len(self.data['train']['ent2id'])
        args.num_rel = len(self.data['train']['rel2id'])
        args.num_time = len(self.data['train']['time2id'])

        # dataset for validation and testing
        self.valid_data = ValidData(args, self.data['valid'])
        self.test_data = TestData(args, self.data['test'])

        # kge models
        self.kge_model = KGEModel(args).to(args.gpu)

        # optimizer
        self.optimizer = None

        # args for controlling training
        self.num_step = None
        self.log_per_step = None
        self.check_per_step = None
        self.early_stop_patience = None

    def write_training_loss(self, loss, step):
        self.writer.add_scalar("training/loss", loss, step)

    def write_evaluation_result(self, results, e):
        self.writer.add_scalar("evaluation/mrr", results['mrr'], e)
        self.writer.add_scalar("evaluation/hits10", results['hits@10'], e)
        self.writer.add_scalar("evaluation/hits5", results['hits@5'], e)
        self.writer.add_scalar("evaluation/hits1", results['hits@1'], e)

    def write_rst_csv(self, suffix_dict, query_part):
        for suf, rst in suffix_dict.items():
            with open(os.path.join(self.args.log_dir, f"{self.args.task_name}_{suf}_{query_part}.csv"), "a") as rstfile:
                rst_writer = csv.writer(rstfile)
                rst_writer.writerow([self.name, round(rst["mrr"], 4), round(rst["hits@1"], 4),
                                     round(rst["hits@5"], 4), round(rst["hits@10"], 4)])

    def save_checkpoint(self, e, state):
        # delete previous checkpoint
        for filename in os.listdir(self.state_path):
            if self.name in filename.split('.') and os.path.isfile(os.path.join(self.state_path, filename)):
                os.remove(os.path.join(self.state_path, filename))
        # save checkpoint
        torch.save(state, os.path.join(self.args.state_dir, self.name,
                                       self.name + '.' + str(e) + '.ckpt'))

    def save_model(self, best_step):
        os.rename(os.path.join(self.state_path, self.name + '.' + str(best_step) + '.ckpt'),
                  os.path.join(self.state_path, self.name + '.best'))

    def get_curr_state(self):
        raise NotImplementedError

    def before_test_load(self):
        raise NotImplementedError

    def train_one_step(self):
        raise NotImplementedError

    def train(self):
        best_step = 0
        best_eval_rst = {'mrr': 0, 'hits@1': 0, 'hits@5': 0, 'hits@10': 0}
        bad_count = 0
        self.logger.info('start training')

        for i in range(1, self.num_step + 1):
            loss = self.train_one_step()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.log_per_step == 0:
                self.logger.info('step: {} | loss: {:.4f}'.format(i, loss.item()))
                self.write_training_loss(loss.item(), i)

            if i % self.check_per_step == 0 or i == 1:
                eval_rst = self.evaluate()
                self.write_evaluation_result(eval_rst, i)

                if eval_rst['mrr'] > best_eval_rst['mrr']:
                    best_eval_rst = eval_rst
                    best_step = i
                    self.logger.info('best model | mrr {:.4f}'.format(best_eval_rst['mrr']))
                    self.save_checkpoint(i, self.get_curr_state())
                    bad_count = 0
                else:
                    bad_count += 1
                    self.logger.info('best model is at step {0}, mrr {1:.4f}, bad count {2}'.format(
                        best_step, best_eval_rst['mrr'], bad_count))

            if bad_count >= self.early_stop_patience:
                self.logger.info('early stop at step {}'.format(i))
                break

        self.logger.info('finish training')
        self.logger.info('save best model')
        self.save_model(best_step)

        self.logger.info('best validation | mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            best_eval_rst['mrr'], best_eval_rst['hits@1'],
            best_eval_rst['hits@5'], best_eval_rst['hits@10']))

        self.before_test_load()
        rst_all, rst_all_dict = self.evaluate(istest=True)
        rst_50, rst_50_dict = self.evaluate(istest=True, num_cand=50)

        self.write_rst_csv({'all': rst_all, '50': rst_50}, 'all_query')

        for k, v in rst_all_dict.items():
            self.write_rst_csv({'all': v}, k)

        for k, v in rst_50_dict.items():
            self.write_rst_csv({'50': v}, k)

    def get_eval_emb(self, eval_data):
        raise NotImplementedError

    def evaluate(self, istest=False, num_cand='all'):
        if not istest:
            eval_dataloader = DataLoader(EvalDataset(self.args, self.valid_data, self.valid_data.que_triples),
                                         batch_size=self.args.eval_bs,
                                         num_workers=max(1, self.args.cpu_num),
                                         collate_fn=EvalDataset.collate_fn)

            eval_dataloader.dataset.num_cand = num_cand

            ent_emb, rel_emb, time_emb = self.get_eval_emb(self.valid_data)

            results, count = self.get_rank(eval_dataloader, ent_emb, rel_emb, time_emb, num_cand)

            for k, v in results.items():
                results[k] = v / count

            self.logger.info('{} | mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
                num_cand,
                results['mrr'], results['hits@1'],
                results['hits@5'], results['hits@10']))

            return results
        else:
            ent_emb, rel_emb, time_emb = self.get_eval_emb(self.test_data)

            uent_dataloader = DataLoader(EvalDataset(self.args, self.test_data, self.test_data.que_uent),
                                         batch_size=self.args.eval_bs,
                                         num_workers=max(1, self.args.cpu_num),
                                         collate_fn=EvalDataset.collate_fn)
            uent_dataloader.dataset.num_cand = num_cand

            urel_dataloader = DataLoader(EvalDataset(self.args, self.test_data, self.test_data.que_urel),
                                         batch_size=self.args.eval_bs,
                                         num_workers=max(1, self.args.cpu_num),
                                         collate_fn=EvalDataset.collate_fn)
            urel_dataloader.dataset.num_cand = num_cand

            utime_dataloader = DataLoader(EvalDataset(self.args, self.test_data, self.test_data.que_utime),
                                         batch_size=self.args.eval_bs,
                                         num_workers=max(1, self.args.cpu_num),
                                         collate_fn=EvalDataset.collate_fn)
            utime_dataloader.dataset.num_cand = num_cand

            uentrel_dataloader = DataLoader(EvalDataset(self.args, self.test_data, self.test_data.que_uentrel),
                                          batch_size=self.args.eval_bs,
                                          num_workers=max(1, self.args.cpu_num),
                                          collate_fn=EvalDataset.collate_fn)
            uentrel_dataloader.dataset.num_cand = num_cand

            uenttime_dataloader = DataLoader(EvalDataset(self.args, self.test_data, self.test_data.que_uenttime),
                                          batch_size=self.args.eval_bs,
                                          num_workers=max(1, self.args.cpu_num),
                                          collate_fn=EvalDataset.collate_fn)
            uenttime_dataloader.dataset.num_cand = num_cand

            ureltime_dataloader = DataLoader(EvalDataset(self.args, self.test_data, self.test_data.que_ureltime),
                                          batch_size=self.args.eval_bs,
                                          num_workers=max(1, self.args.cpu_num),
                                          collate_fn=EvalDataset.collate_fn)
            ureltime_dataloader.dataset.num_cand = num_cand

            uall_dataloader = DataLoader(EvalDataset(self.args, self.test_data, self.test_data.que_uall),
                                         batch_size=self.args.eval_bs,
                                         num_workers=max(1, self.args.cpu_num),
                                         collate_fn=EvalDataset.collate_fn)
            uall_dataloader.dataset.num_cand = num_cand

            uent_results, uent_count = self.get_rank2(uent_dataloader, ent_emb, rel_emb, time_emb, num_cand, "uent")
            urel_results, urel_count = self.get_rank2(urel_dataloader, ent_emb, rel_emb, time_emb, num_cand, "urel")
            utime_results, utime_count = self.get_rank2(utime_dataloader, ent_emb, rel_emb, time_emb, num_cand, "utime")
            uentrel_results, uentrel_count = self.get_rank2(uentrel_dataloader, ent_emb, rel_emb, time_emb, num_cand, "uentrel")
            uenttime_results, uenttime_count = self.get_rank2(uenttime_dataloader, ent_emb, rel_emb, time_emb, num_cand, "uenttime")
            ureltime_results, ureltime_count = self.get_rank2(ureltime_dataloader, ent_emb, rel_emb, time_emb, num_cand, "ureltime")
            uall_results, uall_count = self.get_rank2(uall_dataloader, ent_emb, rel_emb, time_emb, num_cand, "uall")

            # uent_results, uent_count = self.get_rank(uent_dataloader, ent_emb, rel_emb, time_emb, num_cand)
            # urel_results, urel_count = self.get_rank(urel_dataloader, ent_emb, rel_emb, time_emb, num_cand)
            # utime_results, utime_count = self.get_rank(utime_dataloader, ent_emb, rel_emb, time_emb, num_cand)
            # uentrel_results, uentrel_count = self.get_rank(uentrel_dataloader, ent_emb, rel_emb, time_emb, num_cand)
            # uenttime_results, uenttime_count = self.get_rank(uenttime_dataloader, ent_emb, rel_emb, time_emb, num_cand)
            # ureltime_results, ureltime_count = self.get_rank(ureltime_dataloader, ent_emb, rel_emb, time_emb, num_cand)
            # uall_results, uall_count = self.get_rank(uall_dataloader, ent_emb, rel_emb, time_emb, num_cand)

            results = ddict()
            for k in uent_results.keys():
                results[k] = (uent_results[k] + urel_results[k] + utime_results[k] + uentrel_results[k] + uenttime_results[k] + ureltime_results[k]+ uall_results[k]) / (uent_count + urel_count + utime_count + uentrel_count +uenttime_count + ureltime_count+ uall_count)

            for k, v in uent_results.items():
                uent_results[k] = v / uent_count

            for k, v in urel_results.items():
                urel_results[k] = v / urel_count

            for k, v in utime_results.items():
                utime_results[k] = v / utime_count

            for k, v in uentrel_results.items():
                uentrel_results[k] = v / uentrel_count
            for k, v in uenttime_results.items():
                uenttime_results[k] = v / uenttime_count
            for k, v in ureltime_results.items():
                ureltime_results[k] = v / ureltime_count

            for k, v in uall_results.items():
                uall_results[k] = v / uall_count

            self.logger.info('{} | mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
                num_cand,
                results['mrr'], results['hits@1'],
                results['hits@5'], results['hits@10']))

            return results, {'uent': uent_results, 'urel': urel_results, 'utime': utime_results, 'uentrel': uentrel_results, 'uenttime': uenttime_results, 'ureltime': ureltime_results, 'uall': uall_results}

    def get_rank(self, eval_dataloader, ent_emb, rel_emb, time_emb, num_cand='all'):
        results = ddict(float)
        count = 0

        if num_cand == 'all':
            for batch in eval_dataloader:
                pos_triple, tail_label, head_label = [b.to(self.args.gpu) for b in batch]
                head_idx, rel_idx, tail_idx = pos_triple[:, 0], pos_triple[:, 1], pos_triple[:, 2]

                # tail prediction
                pred = self.kge_model((pos_triple, None), ent_emb, rel_emb, time_emb, mode='tail-batch')

                b_range = torch.arange(pred.size()[0], device=self.args.gpu)
                target_pred = pred[b_range, tail_idx]
                pred = torch.where(tail_label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, tail_idx] = target_pred

                tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                               dim=1, descending=False)[b_range, tail_idx]

                # head prediction
                pred = self.kge_model((pos_triple, None), ent_emb, rel_emb, time_emb, mode='head-batch')

                b_range = torch.arange(pred.size()[0], device=self.args.gpu)
                target_pred = pred[b_range, head_idx]
                pred = torch.where(head_label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, head_idx] = target_pred

                head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                               dim=1, descending=False)[b_range, head_idx]

                ranks = torch.cat([tail_ranks, head_ranks])
                ranks = ranks.float()
                count += torch.numel(ranks)
                results['mr'] += torch.sum(ranks).item()
                results['mrr'] += torch.sum(1.0 / ranks).item()

                for k in [1, 5, 10]:
                    results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])
        else:
            for i in range(self.args.num_sample_cand):
                for batch in eval_dataloader:
                    pos_triple, tail_cand, head_cand = [b.to(self.args.gpu) for b in batch]

                    b_range = torch.arange(pos_triple.size()[0], device=self.args.gpu)
                    target_idx = torch.zeros(pos_triple.size()[0], device=self.args.gpu, dtype=torch.int64) + num_cand

                    # tail prediction
                    pred = self.kge_model((pos_triple, tail_cand), ent_emb, rel_emb, time_emb, mode='tail-batch')
                    tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                                   dim=1, descending=False)[b_range, target_idx]
                    # head prediction
                    pred = self.kge_model((pos_triple, head_cand), ent_emb, rel_emb, time_emb, mode='head-batch')
                    head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                                   dim=1, descending=False)[b_range, target_idx]

                    ranks = torch.cat([tail_ranks, head_ranks])
                    ranks = ranks.float()
                    count += torch.numel(ranks)
                    results['mr'] += torch.sum(ranks).item()
                    results['mrr'] += torch.sum(1.0 / ranks).item()

                    for k in [1, 5, 10]:
                        results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

        return results, count

    def get_rank2(self, eval_dataloader, ent_emb, rel_emb, time_emb, num_cand='all', test_mode="uent"):
        results = ddict(float)
        count = 0

        if num_cand == 'all':
            for batch in eval_dataloader:
                pos_triple, tail_label, head_label = [b.to(self.args.gpu) for b in batch]
                head_idx, rel_idx, tail_idx = pos_triple[:, 0], pos_triple[:, 1], pos_triple[:, 2]

                # tail prediction
                pred = self.kge_model((pos_triple, None), ent_emb, rel_emb, time_emb, mode='tail-batch')

                b_range = torch.arange(pred.size()[0], device=self.args.gpu)
                target_pred = pred[b_range, tail_idx]
                pred = torch.where(tail_label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, tail_idx] = target_pred

                tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                               dim=1, descending=False)[b_range, tail_idx]

                # head prediction
                pred = self.kge_model((pos_triple, None), ent_emb, rel_emb, time_emb, mode='head-batch')

                b_range = torch.arange(pred.size()[0], device=self.args.gpu)
                target_pred = pred[b_range, head_idx]
                pred = torch.where(head_label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, head_idx] = target_pred

                head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                               dim=1, descending=False)[b_range, head_idx]

                ranks = torch.cat([tail_ranks, head_ranks])
                ranks = ranks.float()
                count += torch.numel(ranks)
                results['mr'] += torch.sum(ranks).item()
                results['mrr'] += torch.sum(1.0 / ranks).item()

                for k in [1, 5, 10]:
                    results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])
        else:
            f_head = open('./ranks/{}_{}_head'.format(self.args.task_name, test_mode), 'w')
            f_tail = open('./ranks/{}_{}_tail'.format(self.args.task_name, test_mode), 'w')
            # 保存三元组针对所有尾实体的排名情况
            f_rank_head = open('./ranks_index_top10/{}_{}_head.csv'.format(self.args.task_name, test_mode), 'w',
                              encoding="UTF-8", newline='')
            writer_head = csv.writer(f_rank_head, delimiter=",")

            f_rank_tail = open('./ranks_index_top10/{}_{}_tail.csv'.format(self.args.task_name, test_mode), 'w',
                               encoding="UTF-8", newline='')
            writer_tail = csv.writer(f_rank_tail, delimiter=",")
            for i in range(self.args.num_sample_cand):
                for batch in eval_dataloader:
                    pos_triple, tail_cand, head_cand = [b.to(self.args.gpu) for b in batch]
                    sub, rel, obj, time = pos_triple[:, 0], pos_triple[:, 1], pos_triple[:, 2], pos_triple[:, 3]
                    b_range = torch.arange(pos_triple.size()[0], device=self.args.gpu)
                    target_idx = torch.zeros(pos_triple.size()[0], device=self.args.gpu, dtype=torch.int64) + num_cand

                    # tail prediction
                    pred = self.kge_model((pos_triple, tail_cand), ent_emb, rel_emb, time_emb, mode='tail-batch')
                    tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                                   dim=1, descending=False)[b_range, target_idx]
                    ####针对所有排名 返回排名前10的索引 前三列对应h r t后10列对应排名前10的索引
                    rank_index = torch.argsort(pred, dim=1, descending=True)[:, :10]
                    t_rank_index = torch.cat([sub.unsqueeze(1), rel.unsqueeze(1), obj.unsqueeze(1), time.unsqueeze(1), tail_cand[torch.arange(tail_cand.size()[0]), rank_index.t()].t()], dim=1)
                    t_rank_index = t_rank_index.detach().cpu().numpy().astype(int)
                    writer_tail.writerows(t_rank_index)
                    ######

                    # head prediction
                    pred = self.kge_model((pos_triple, head_cand), ent_emb, rel_emb, time_emb, mode='head-batch')
                    head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                                   dim=1, descending=False)[b_range, target_idx]
                    ####针对所有排名 返回排名前10的索引 前三列对应h r t后10列对应排名前10的索引
                    rank_index = torch.argsort(pred, dim=1, descending=True)[:, :10]
                    t_rank_index = torch.cat([sub.unsqueeze(1), rel.unsqueeze(1), obj.unsqueeze(1), time.unsqueeze(1), head_cand[torch.arange(head_cand.size()[0]), rank_index.t()].t()], dim=1)
                    t_rank_index = t_rank_index.detach().cpu().numpy().astype(int)
                    writer_head.writerows(t_rank_index)
                    ######

                    ranks = torch.cat([tail_ranks, head_ranks])
                    ranks = ranks.float()
                    for i in range(pos_triple.shape[0]):
                        f_tail.write('{},{},{},{},{}\n'.format(pos_triple[i][0], pos_triple[i][1], pos_triple[i][2], pos_triple[i][3], ranks[i]))
                    for i in range(pos_triple.shape[0]):
                        f_head.write('{},{},{},{},{}\n'.format(pos_triple[i][0], pos_triple[i][1], pos_triple[i][2], pos_triple[i][3], ranks[i+pos_triple.shape[0]]))

                    count += torch.numel(ranks)
                    results['mr'] += torch.sum(ranks).item()
                    results['mrr'] += torch.sum(1.0 / ranks).item()

                    for k in [1, 5, 10]:
                        results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

        return results, count