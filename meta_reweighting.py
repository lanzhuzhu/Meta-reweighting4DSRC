from __future__ import division
from sub_model import *
import os, sys
import time
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import operator
import random
import datetime
from collections import Counter

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("use cuda!")
else:
    device = torch.device("cpu")
from collections import OrderedDict

relation2id = {}
max_length = 80  # 50
max_pos = 161  # 80
warm_up = 0.05


def set_lr(progress):
    if progress < warm_up:
        return progress / warm_up
    return max((progress - 1.) / (warm_up - 1.), 0.)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



def init_relation(pre_processed_dir=''):
    # reading relation ids...
    global relation2id
    print('reading relation ids...')
    with open(pre_processed_dir + "relation2id.txt", "r", encoding='utf-8') as inf:
        for line in inf.readlines():
            content = line.strip().split()
            relation2id[content[0]] = int(content[1])
    return relation2id


def load_data(data_dir, data_flag):
    data_word = np.load(data_dir + data_flag + "_word.npy")
    data_pos1 = np.load(data_dir + data_flag + "_pos1.npy")
    data_pos2 = np.load(data_dir + data_flag + "_pos2.npy")
    data_label = np.load(data_dir + data_flag + "_label.npy")
    data_eposs = np.load(data_dir + data_flag + "_eposs.npy")
    return data_word, data_pos1, data_pos2, data_label, data_eposs


def selected_array_byValues(values, count):
    return np.argsort(values)[-count:][::-1]


def train_lre(data_dir, epochs, directory, Wv, pf1,
              pf2, batch=50, num_classes=6, to_train=1, input_model="", reload_flag=False, enhanced_flag=True):
    """
    epochs: the # iterations
    Wv: the file name of Word Vector data
    batch: batch_size
    num_classes: # classes of the dataset
    to_train: the training flag
    test_epoch: no use param
    """

    lr = 0.1
    start_time = time.time()
    out_dir = directory[0]
    pyfile = directory[1][0]
    split_idx = directory[1][1]
    sp_num = directory[2]

    print("epochs:%d, batch_size:%d, learning_rate:%f" % (epochs, batch, lr))

    if to_train and not reload_flag:
        time_str = datetime.datetime.now().isoformat()

    else:
        time_str = input_model
        print("test %s or reload saved model:" % time_str)

    model_file_dir = out_dir + "saved_model/" + time_str + "/"
    if not os.path.exists(model_file_dir):
        os.makedirs(os.path.dirname(model_file_dir))
    model_file = model_file_dir + "model_epoch_12"

    log_file = out_dir + "logs/"
    if not os.path.exists(log_file):
        os.makedirs(log_file)
    result_dir = out_dir + "results/"
    results_file = out_dir + "results/" + "predictedPRF_splitV" + split_idx + ".txt"
    if not os.path.exists(os.path.dirname(results_file)):
        os.makedirs(os.path.dirname(results_file))

    dev_ratio = 3  # the ratio added to dev0 as how many folds of dev0 size
    test_batch = 1000
    pre_epoch = 1

    metric_id = 1
    metric_name = ["weights", "prob"]
    if metric_id:
        use_prob_flag = True
        decay_meta = False
        discount_factor = 0.97
        start_use_sa = 3  # 12 #
    else:
        use_prob_flag = False
        decay_meta = True
        discount_factor = 1
        start_use_sa = 12

    accumulated_flag = True
    hyper_flag = "Elite_by_" + metric_name[metric_id] if enhanced_flag else "l2rw_"
    hyper_flag += sp_num + "split_" + split_idx + "dsc_" + str(discount_factor) + "pre_epk" + str(pre_epoch) + "st_epk_" + str(start_use_sa) + "K_" + str(
        dev_ratio)
    print(hyper_flag)

    model = PCNN(word_length=len(Wv), feature_length=len(pf1), cnn_layers=230, kernel_size=(3, 360),
                 Wv=Wv, pf1=pf1, pf2=pf2, num_classes=num_classes)
    model.to(device)
    optimizer = optim.SGD(model.params(), lr=lr, weight_decay=1e-5)

    if to_train == 1:
        dev_results = []
        test_results = []
        best_epoch = [0, 0, 0]  # best_dev_f1, best_dev_acc, test_f1,
        best_F1 = 0
        best_acc = 0
        train_data, train_ldists, train_rdists, train_labels, train_eposs = load_data(data_dir, "train")
        dev_data, dev_ldists, dev_rdists, dev_labels, dev_eposs = load_data(data_dir + "processed_" + sp_num + "/",
                                                                            "hyper_dev_v" + str(split_idx))
                                                                                
        #reference data - dev0
        dev0_data, dev0_ldists, dev0_rdists, dev0_labels, dev0_eposs = load_data(data_dir + "processed_" + sp_num + "/",
                                                                                 "dnoise_dev_v" + str(split_idx))
                                                                                     
        test_data, test_ldists, test_rdists, test_labels, test_eposs = load_data(data_dir, "test")
        dev0_type_dist = Counter(dev0_labels).most_common()
        dev0_size = dev0_labels.shape[0]
        print("train sents:", str(len(train_labels)))
        train_size = train_data.shape[0]
        print("origin dev0_types:", dev0_type_dist)
        index_array = np.arange(train_size)

        dev0_type_dist = [(int(i), v) for (i, v) in dev0_type_dist]
        dev0_type_dist.pop(0)

        early_stopping_num = 20  #
        es_Countdown = early_stopping_num

        seleceted2dev0 = []
        min_lr = 0.005

        if reload_flag:
            print("Loading:", model_file, "saved model")
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print("reload lr is", get_lr(optimizer))

        else:
            start_epoch = 0

        if enhanced_flag:
            ins_weights = np.zeros((train_data.shape[0],), dtype="float32")
            accumulated_scores = np.zeros((train_data.shape[0],), dtype="float32")
            rank2score_inrel = {}  # np.log(1+np.exp(-np.arange(50)))
            rel_idx = {}
            ranknum = {}
            for rel, num in dev0_type_dist:
                rel_idx[rel] = np.where(train_labels == rel)[0]  # store the index of index_array
                ranknum[rel] = min(len(rel_idx[rel]), num * 50)
                rank2score_inrel[rel] = 1 / (1 + np.exp(np.arange(
                    ranknum[rel]) - num))

        batch_num = int(np.ceil(train_size / float(batch)))
        total_steps = 20 * batch_num
        steps = start_epoch * batch_num
        for epoch in range(start_epoch, epochs):
            total_loss = 0.
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            if epoch >= pre_epoch and enhanced_flag:
                batches = _make_batches(train_size, test_batch)
                model.eval()
                probb = np.zeros((train_size,), dtype='float32')
                if use_prob_flag:
                    for batch_index, (batch_start, batch_end) in enumerate(batches):
                        batch_ids = index_array[batch_start:batch_end]
                        x_slice = torch.from_numpy(_slice_arrays(train_data, batch_ids)).long().to(device)
                        l_slice = torch.from_numpy(_slice_arrays(train_ldists, batch_ids)).long().to(device)
                        r_slice = torch.from_numpy(_slice_arrays(train_rdists, batch_ids)).long().to(device)
                        e_slice = torch.from_numpy(_slice_arrays(train_eposs, batch_ids)).long().to(device)

                        # put the data into variable
                        x_batch = autograd.Variable(x_slice, requires_grad=False)
                        l_batch = autograd.Variable(l_slice, requires_grad=False)
                        r_batch = autograd.Variable(r_slice, requires_grad=False)
                        e_batch = e_slice
                        results_batch, attention_scores = model(x_batch, l_batch, r_batch, e_batch)
                        all_probab = F.softmax(results_batch, dim=-1).data.cpu().numpy()  # (batch_size, n_class)
                        index = train_labels[batch_ids]
                        probb[batch_start:batch_end] = all_probab[np.arange(len(batch_ids)), index]

                print("org weights range is ", np.max(ins_weights), np.min(ins_weights))
                norm_weights = ins_weights / np.max(ins_weights)
                print("after norm on weights,max and min are:", np.max(norm_weights), np.min(norm_weights))

                if use_prob_flag:
                    print(" before preprocessing probb range is ", np.max(probb), np.min(probb))
                    post_probb = probb / np.sum(probb)
                    norm_probb = post_probb * 10 ** int(np.log10(1 / np.max(post_probb)) + 1)
                    print("after norm  probb range is ", np.max(norm_probb), np.min(norm_probb))
                else:
                    norm_probb = probb
                if metric_id:
                    metrics = [norm_probb]
                else:
                    metrics = [norm_weights]

                selected_idx = OrderedDict()
                accumulated_scores *= discount_factor
                for rel, count in dev0_type_dist:
                    count = int(count * dev_ratio)
                    top_idx_inrel = []
                    for cur_score in metrics:
                        cur_score_inrel = cur_score[rel_idx[rel]]
                        idx2sa = cur_score_inrel.argsort()[::-1][:ranknum[rel]]  # idx in rel from top to bottom
                        top_idx_inrel.append(idx2sa[:count])
                        if accumulated_flag:
                            accumulated_scores[rel_idx[rel][idx2sa]] += rank2score_inrel[rel][:ranknum[rel]]
                    if accumulated_flag and epoch >= start_use_sa:
                        top_idx_inrel = np.argsort(accumulated_scores[rel_idx[rel]])[-count:][::-1]
                        print("rel : final selected index's weights :\t" + str(rel) + "\t" + str(
                            norm_weights[rel_idx[rel][top_idx_inrel]]) + "\t and accumulated scores: \t" + str(
                            accumulated_scores[rel_idx[rel][top_idx_inrel]]) + "\n")
                    else:  # assume no more than 2 metrics
                        top_idx_inrel = top_idx_inrel[0]
                    selected_idx[rel] = rel_idx[rel][top_idx_inrel]
                seleceted2dev0 = np.concatenate(list(selected_idx.values()))
                print("seleceted2dev0 shape", seleceted2dev0.shape)


            batches = _make_batches(train_size, batch)
            random.shuffle(index_array)
            print(str(now), "\tStarting Epoch", (epoch), "\tBatches:", len(batches))

            model.train()
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                x_slice = torch.from_numpy(_slice_arrays(train_data, batch_ids)).long().to(device)
                l_slice = torch.from_numpy(_slice_arrays(train_ldists, batch_ids)).long().to(device)
                r_slice = torch.from_numpy(_slice_arrays(train_rdists, batch_ids)).long().to(device)
                e_slice = torch.from_numpy(_slice_arrays(train_eposs, batch_ids)).long().to(device)
                train_labels_slice = torch.from_numpy(_slice_arrays(train_labels, batch_ids)).long().to(device)
                # put the data into variable
                x_batch = autograd.Variable(x_slice, requires_grad=False)
                l_batch = autograd.Variable(l_slice, requires_grad=False)
                r_batch = autograd.Variable(r_slice, requires_grad=False)
                e_batch = e_slice
                train_labels_batch = autograd.Variable(train_labels_slice, requires_grad=False)  # .squeeze(1)
                # initialize a dummy network for the meta learning of the weights
                meta_model = PCNN(word_length=len(Wv), feature_length=len(pf1), cnn_layers=230, kernel_size=(3, 360),
                                  Wv=Wv, pf1=pf1, pf2=pf2, num_classes=num_classes)
                if torch.cuda.is_available():
                    meta_model.to(device)
                meta_model.load_state_dict(model.state_dict())
                #  initial forward pass to compute the initial weighted loss
                results_batch, attention_scores = meta_model(x_batch, l_batch, r_batch,
                                                             e_batch)

                # put the dev data into variable  # it means class the forward function.
                loss = F.cross_entropy(results_batch, train_labels_batch,
                                       reduce=False)  # return each instance loss without reduction,minimize
                eps = nn.Parameter(torch.zeros(loss.size()).to(device))  # then included in meta_model.params();
                l_f_meta = torch.sum(loss * eps)  # lf = sigma(eps*loss)
                meta_model.zero_grad()
                #  perform a parameter update
                grads = torch.autograd.grad(l_f_meta, (meta_model.params()),
                                            create_graph=True)  # without normalization, in fact it is 0
                meta_model.update_params(0.1, source_params=grads)
                # 2nd forward pass and getting the gradients with respect to epsilon
                meta_model.eval()
                # prepare the reference data

                dev_data_s = dev0_data
                dev_labels_s = dev0_labels
                dev_ldists_s = dev0_ldists
                dev_rdists_s = dev0_rdists
                dev_eposs_s = dev0_eposs

                x_dev = autograd.Variable(torch.from_numpy(dev_data_s).long().to(device), requires_grad=False)
                l_dev = autograd.Variable(torch.from_numpy(dev_ldists_s).long().to(device), requires_grad=False)
                r_dev = autograd.Variable(torch.from_numpy(dev_rdists_s).long().to(device), requires_grad=False)
                e_dev = torch.from_numpy(dev_eposs_s).long().to(device)
                dev_labels_s = autograd.Variable(torch.from_numpy(dev_labels_s).long().to(device),
                                                 requires_grad=False)  # .squeeze(1)
                # get the result
                results_dev, attention_dev = meta_model(x_dev, l_dev, r_dev, e_dev)
                l_g_meta = F.cross_entropy(results_dev, dev_labels_s)
                grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True, allow_unused=True, retain_graph=True)[
                    0]  # eps not used to compute outputs
                # Line 11 computing and normalizing the weights
                w_tilde = torch.clamp(-grad_eps, min=0)
                norm_c = torch.sum(w_tilde)
                # calculate the weight
                if norm_c != 0:
                    w_clean = w_tilde / norm_c
                else:
                    w_clean = w_tilde
                if epoch >= pre_epoch and enhanced_flag and len(seleceted2dev0) > 0:
                    tsampled_dices = seleceted2dev0
                    dev_data_s = train_data[tsampled_dices]
                    dev_labels_s = train_labels[tsampled_dices]
                    dev_ldists_s = train_ldists[tsampled_dices]
                    dev_rdists_s = train_rdists[tsampled_dices]
                    dev_eposs_s = train_eposs[tsampled_dices]

                    x_dev = autograd.Variable(torch.from_numpy(dev_data_s).long().to(device), requires_grad=False)
                    l_dev = autograd.Variable(torch.from_numpy(dev_ldists_s).long().to(device), requires_grad=False)
                    r_dev = autograd.Variable(torch.from_numpy(dev_rdists_s).long().to(device), requires_grad=False)
                    e_dev = torch.from_numpy(dev_eposs_s).long().to(device)
                    dev_labels_s = autograd.Variable(torch.from_numpy(dev_labels_s).long().to(device),
                                                     requires_grad=False)  # .squeeze(1)
                    # get the result
                    results_dev, attention_dev = meta_model(x_dev, l_dev, r_dev, e_dev)
                    l_g_meta_noisy = F.cross_entropy(results_dev, dev_labels_s)
                    equal_w_noise = len(seleceted2dev0) / (len(seleceted2dev0) + dev0_size)
                    if epoch >= start_use_sa and decay_meta:
                        w_noise = 0.1
                    else:
                        w_noise = equal_w_noise
                    l_g_meta = (1 - w_noise) * l_g_meta + w_noise * l_g_meta_noisy

                    grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True, allow_unused=True)[0]
                    w_tilde = torch.clamp(-grad_eps, min=0)
                    norm_c = torch.sum(w_tilde)
                    # calculate the weight
                    if norm_c != 0:
                        w = w_tilde / norm_c
                    else:
                        w = w_tilde
                else:
                    w = w_clean

                model.train()
                results_batch, attention_scores = model(x_batch, l_batch, r_batch, e_batch)
                loss = F.cross_entropy(results_batch, train_labels_batch,
                                       reduce=False)  # when use weight for instance the reduce is False, that means evrey instance loss is returned.

                l_f = torch.sum(loss * w)
                if enhanced_flag:
                    ins_weights[batch_ids] = w_clean.data.cpu().numpy()
                    # ins_weights[batch_ids] = w.data.cpu().numpy()

                total_loss += l_f.data
                # backprop
                optimizer.zero_grad()
                l_f.backward()
                steps += 1
                optimizer.param_groups[0]['lr'] = max(lr * set_lr(steps / total_steps), min_lr)
                optimizer.step()
                if batch_index % 100 == 0 or batch_index == len(batches) - 1:
                    print("epoch {} batch {} training loss: {}".format(epoch + 1, batch_index + 1, l_f.data))

            # validate part
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print(str(now), "\tDone Epoch", (epoch + 1), "\nLoss:", total_loss)

            print("Now validation starts")

            # calcualate the precision and recall
            dev_pr, dev_rel_pr = get_test_pr(dev_data, dev_labels, dev_ldists, dev_rdists, dev_eposs, num_classes,
                                             model, batch)
            dev_results.append(dev_pr)
            test_pr, test_rel_pr = get_test_pr(test_data, test_labels, test_ldists, test_rdists, test_eposs,
                                               num_classes, model, batch)
            test_results.append(test_pr)

            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print(str(now) + '\t epoch ' + str(epoch) + "\tValidate Precision : " + str(
                dev_pr[0]) + "\t Recall: " + str(dev_pr[1]) + "\t F1score: " + str(dev_pr[2]) + '\n')

            f_log = open(results_file, 'a+', 1)
            if epoch == 0:
                f_log.write("this running start time:" + str(start_time) + "\t model_flag: " + str(
                    time_str) + '\t split :' + str(split_idx) + '\n')
            f_log.write(str(now) + '\t epoch ' + str(epoch) + "; lr:" + str(get_lr(optimizer)) + "\ttime_stamp\t" + str(
                time_str) + '\n')
            f_log.write('--------------dev--------------------\n')
            for k, v in dev_rel_pr.items():
                f_log.write(
                    "dev relationID: P : R: F1: \t" + str(k) + "\t" + str(v['P']) + "\t" + str(v['R']) + "\t" + str(
                        v['F']) + '\n')
            f_log.write('--------------test--------------------\n')
            for k, v in test_rel_pr.items():
                f_log.write("test relationID: P : R: F1: \t" + str(k) + "\t" + str(v['P']) + "\t" + str(
                    v['R']) + "\t" + str(v['F']) + '\n')
            f_log.close()

            if reload_flag and epoch == start_epoch:
                model_file_dir = model_file_dir + "reload/"
                if not os.path.exists(model_file_dir):
                    os.makedirs(os.path.dirname(model_file_dir))
            if dev_pr[3] > best_acc:
                best_acc = dev_pr[3]
                best_epoch[1] = epoch
                es_Countdown = early_stopping_num
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                           model_file_dir + "best_dev_acc")

            if dev_pr[2] > best_F1:
                best_F1 = dev_pr[2]
                best_epoch[0] = epoch
                es_Countdown = early_stopping_num
                print("saving the best f1 model....")
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                           model_file_dir + "best_dev_f1")

            es_Countdown -= 1
            if es_Countdown == 0:
                print("early stopping at epoch:", epoch)
                early_stopping_num = 100  # continue training until enough
            if early_stopping_num == 100 and epoch >= 20:
                break
        # write results to log files
        f_dev_log = open(log_file + 'valid_' + split_idx + '_log.txt', 'a+', 1)
        f_test_log = open(log_file + 'test_' + split_idx + '_log.txt', 'a+', 1)
        best_test_f1_tuple = test_results[0]
        assert len(dev_results) == len(test_results)
        for i in range(len(dev_results)):
            if i == 0:
                if reload_flag:
                    f_dev_log.write('reload:\t' + str(time_str) + '\t start at:' + str(start_time) + '\n')
                    f_test_log.write('reload:\t' + str(time_str) + '\t start at:' + str(start_time) + '\n')

                else:
                    f_dev_log.write(str(time_str) + '\n')
                    f_test_log.write(str(time_str) + '\n')
            f_dev_log.write(
                'epoch ' + str(i + start_epoch) + "\t Precision \t Recall\tF1score:\tAcc " + str(
                    dev_results[i][0]) + "\t" + str(
                    dev_results[i][1]) + "\t" + str(dev_results[i][2]) + "\t" + str(dev_results[i][3]) + '\n')
            f_test_log.write(
                'epoch ' + str(i + start_epoch) + "\t P : R: F1:ACC\t" + str(test_results[i][0]) + "\t " + str(
                    test_results[i][1]) + "\t " + str(test_results[i][2]) + "\t " + str(test_results[i][3]) + '\n')
            if test_results[i][2] > best_test_f1_tuple[2]:
                best_test_f1_tuple = test_results[i]
                best_epoch[2] = i
        f_dev_log.close()
        f_test_log.close()
        best_test_idx = 0 if test_results[best_epoch[0] - start_epoch][2] > test_results[best_epoch[1] - start_epoch][
            2] else 1

        print("best dev f1,test f1 at epoch %d:%d" % (best_epoch[0], best_epoch[1]))
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        fi_log = open(log_file + 'final_log.txt', 'a+', 1)

        print(str(now) + "\t" + str(time_str) + "\t" + "\t".join(
            [str(j) for j in test_results[best_epoch[best_test_idx] - start_epoch]]) + "\t" + "\tEpochs:" + str(
            epochs) + ",batch_size:" + str(batch) + " lr:" + str(
            lr) + pyfile + "; split_v" + split_idx + "; sampledRef: " + sp_num + ";\t" + "\t selected test at: " + str(
            best_epoch[best_test_idx]) + "best dev f1 , acc epoch:" + str(best_epoch[0]) + "," + str(
            best_epoch[1]) + "\t corresponds dev:\t" + str(
            dev_results[best_epoch[best_test_idx] - start_epoch]) + "\t" + "\t\t\t best test f1 at epoch: " + str(
            best_epoch[2]) + ": " + " ".join([str(i) for i in
                                              best_test_f1_tuple]) + "\t" + "\t\t\t\t\t\t\t training time(hour):training epochs=" + str(
            (time.time() - start_time) / 3600) + ", " + str(len(dev_results)) + "\n")

        fi_log.write(str(time_str) + "\t" + "\t".join(
            [str(j) for j in test_results[best_epoch[best_test_idx] - start_epoch]]) + "\t" + "\tEpochs:" + str(
            epochs) + ",batch_size:" + str(batch) + " lr:" + str(
            lr) + pyfile + "; split_v" + split_idx + " reload:" + str(
            reload_flag) + "\t" + "\t selected test at: " + str(
            best_epoch[best_test_idx]) + "best dev f1 , acc epoch:" + str(best_epoch[0]) + "," + str(
            best_epoch[1]) + "\t corresponds dev:\t" + str(
            dev_results[best_epoch[best_test_idx] - start_epoch]) + "\t" + "\t\t\t best test f1 at epoch: " + str(
            best_epoch[2]) + ": " + " ".join([str(i) for i in
                                              best_test_f1_tuple]) + "\t" + "\t\t\t\t\t\t\t training time(hour):training epochs=" + str(
            (time.time() - start_time) / 3600) + ", " + str(len(dev_results)) + "\n")

    else:
        # test_model
        model_file_path = out_dir + time_str + "_saved_model/"
        for f_i in ['best_dev_acc', 'best_dev_f1']:  # 'best_dev_acc', 'best_dev_f1',
            model_file = model_file_path + f_i
            if os.path.exists(model_file):
                print("Loading:", model_file, "saved model")
                checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                model.eval()
                print("start testing")
                test_data, test_ldists, test_rdists, test_labels, test_eposs = load_data(data_dir, "test")
                dev_data, dev_ldists, dev_rdists, dev_labels, dev_eposs = load_data(data_dir, "dev")
                test_pr, test_rel_pr = get_test_pr(test_data, test_labels, test_ldists, test_rdists, test_eposs,
                                                   num_classes,
                                                   model,
                                                   batch)
                dev_pr, dev_rel_pr = get_test_pr(dev_data, dev_labels, dev_ldists, dev_rdists, dev_eposs, num_classes,
                                                 model, batch)

                print('dev', dev_pr)
                print('test', test_pr)

                now = time.strftime("%Y-%m-%d %H:%M:%S")
                result_file = result_dir + 'test_relation.txt'
                if not os.path.exists(os.path.dirname(result_file)):
                    os.makedirs(os.path.dirname(result_file))
                f_log = open(result_file, 'a+', 1)
                f_log.write(str(now) + "\tTest model:" + str(model_file) + '\n')
                f_log.write('--------------dev--------------------\n')
                for k, v in dev_rel_pr.items():
                    f_log.write(
                        "dev relationID: P : R: F1: \t" + str(k) + "\t" + str(v['P']) + "\t" + str(v['R']) + "\t" + str(
                            v['F']) + '\n')
                f_log.write('--------------test--------------------\n')
                for k, v in test_rel_pr.items():
                    f_log.write("test relationID: P : R: F1: \t" + str(k) + "\t" + str(v['P']) + "\t" + str(
                        v['R']) + "\t" + str(v['F']) + '\n')

                f_log.close()


def pr(predict_y, true_y):
    """
    predict_y: (#instance, )
    true_y: (#instance, 1)
    """

    total = sum([1 for y_i in true_y if y_i])

    t_p = 0.0  # true postive
    f_p = 0.0  # false postive
    t_n = 0.0
    f_n = 0.0  # false negative

    for real, pred in zip(true_y, predict_y):
        if pred == 0:
            if real == 0:
                t_n += 1  # true negative
            else:
                f_n += 1  # false negative
        else:
            if pred == real:
                t_p += 1  # true postive
            else:
                f_p += 1  # false positve

    try:
        prec = t_p / (t_p + f_p)
    except:
        prec = 1.0
    rec = t_p / total if total else 0  # recall = (relevant)/total
    try:
        f1 = 2 * prec * rec / (prec + rec)
    except:
        f1 = 0

    acc = (t_p + t_n) / (len(true_y))
    return [prec, rec, f1, acc]


def rel_pr(predict_y, true_y):
    """
    predict_y: (#instance, )
    true_y: (#instance, 1)
    """
    rel_scores = {}
    gold_rels = Counter(true_y).most_common()
    total_correct = 0
    for rel, _ in gold_rels:
        rel_count = 0
        pre_count = 0
        pre_true_count = 0
        for real, pred in zip(true_y, predict_y):
            if real == rel:
                rel_count += 1
            if pred == rel:
                pre_count += 1
            if real == rel and pred == rel:
                pre_true_count += 1
        if rel:  # NA is 0
            total_correct += pre_true_count
        P = pre_true_count / pre_count if pre_count else 0
        R = pre_true_count / rel_count if rel_count else 0
        F = 2 * P * R / (P + R) if (P + R) else 0
        rel_scores[str(rel)] = {'P': P, 'R': R, 'F': F}
    predict_count = sum([1 if p else 0 for p in predict_y])
    relation_count = sum([1 if p else 0 for p in true_y])
    P = total_correct / predict_count if predict_count else 0
    R = total_correct / relation_count if relation_count else 0
    F = 2 * P * R / (P + R) if (P + R) else 0
    macro_p = sum([rel_scores[key]['P'] for key in rel_scores.keys()]) / len(rel_scores.keys())
    macro_r = sum([rel_scores[key]['R'] for key in rel_scores.keys()]) / len(rel_scores.keys())
    macro_f1 = sum([rel_scores[key]['F'] for key in rel_scores.keys()]) / len(rel_scores.keys())
    rel_scores['macro'] = {'P': macro_p, 'R': macro_r, 'F': macro_f1}
    rel_scores['all'] = {'P': P, 'R': R, 'F': F}
    return rel_scores


def _make_batches(size, batch_size):
    num_batches = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, num_batches)]


def _slice_arrays(arrays, start=None, stop=None):
    if isinstance(arrays, list):
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] for x in arrays]
        else:
            return [x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        else:
            return arrays[start:stop]


def get_test_pr(dev_data, dev_labels, dev_ldists, dev_rdists, dev_eposs, numClasses, model, batch):
    samples = dev_data.shape[0]
    batches = _make_batches(samples, batch)
    index_array = np.arange(samples)
    random.shuffle(index_array)
    results = np.zeros((samples, numClasses), dtype='float32')
    labels = np.zeros((samples,), dtype='float32')
    model.eval()

    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        x_slice = torch.from_numpy(_slice_arrays(dev_data, batch_ids)).long().to(device)
        l_slice = torch.from_numpy(_slice_arrays(dev_ldists, batch_ids)).long().to(device)
        r_slice = torch.from_numpy(_slice_arrays(dev_rdists, batch_ids)).long().to(device)
        e_slice = torch.from_numpy(_slice_arrays(dev_eposs, batch_ids)).long().to(device)
        dev_labels_slice = torch.from_numpy(_slice_arrays(dev_labels, batch_ids)).long().to(device)
        # put the data into variable
        x_batch = autograd.Variable(x_slice, requires_grad=False)
        l_batch = autograd.Variable(l_slice, requires_grad=False)
        r_batch = autograd.Variable(r_slice, requires_grad=False)
        e_batch = e_slice
        dev_labels_batch = autograd.Variable(dev_labels_slice, requires_grad=False)  # .squeeze(1)
        results_batch, attention_scores = model(x_batch, l_batch, r_batch, e_batch)
        results[batch_start:batch_end, :] = F.softmax(results_batch, dim=-1).data.cpu().numpy()
        labels[batch_start:batch_end] = dev_labels_batch.data.cpu().numpy()

    # predict_y_dist = np.asarray(np.copy(results))
    rel_type_arr = np.argmax(results, axis=-1)
    dev_pr = pr(rel_type_arr, labels)  # outdated
    dev_rel_pr = rel_pr(rel_type_arr, labels)
    return dev_pr, dev_rel_pr  # ,p_l, r_l


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please enter the arguments correctly! python file.py file_split_index(sp_num)")
        print(len(sys.argv))
        sys.exit()
    elif len(sys.argv) == 3:
        sp_num = sys.argv[2]  # "100"
    else:
        sp_num = "100"  #
    print("sys.argv", sys.argv)

    enhanced_flag = True # use the original meta-reweighting or enhanced version by including elite data

    data_dir = "./kbp_extend_data/"
    out_dir = "./Results/sampled_" + sp_num + "/" 

    seeds = [74, 863, 67920, 5340, 1245] 

    seed = seeds[int(sys.argv[1])]
    print("index and seed are:", sys.argv[1], seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    print('result dir=' + out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    print('load Wv ...')
    Wv = np.load(data_dir + "embedding.npy")

    PF1 = np.asarray(np.random.uniform(low=-1, high=1, size=[max_pos + 1, 30]), dtype='float32')
    padPF1 = np.zeros((1, 30))
    PF1 = np.vstack((padPF1, PF1))
    PF2 = np.asarray(np.random.uniform(low=-1, high=1, size=[max_pos + 1, 30]), dtype='float32')
    padPF2 = np.zeros((1, 30))
    PF2 = np.vstack((padPF2, PF2))
    print("PF1", PF1[5])
    print("PF2", PF2[5])
    time_str = "2020-01-16T06:28:32.773223"
    train_lre(data_dir,
              25,
              (out_dir, sys.argv, sp_num),
              Wv,
              PF1,
              PF2, batch=160, to_train=1, num_classes=6, input_model=time_str, reload_flag=False, enhanced_flag=enhanced_flag)
