import torch
import numpy as np
from datetime import datetime

from spodernet.utils.global_config import Config
from spodernet.utils.cuda_utils import CUDATimer
from spodernet.utils.logger import Logger
from torch.autograd import Variable
from sklearn import metrics

#timer = CUDATimer()
log = Logger('evaluation{0}.py.txt'.format(datetime.now()))

def ranking_and_hits(model, dev_rank_batcher, vocab, name, model_name, literal_representation):

    log.info('')
    log.info('-'*50)
    log.info(name)
    log.info('-'*50)
    log.info('')
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []
    triples = []
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for i, str2var in enumerate(dev_rank_batcher):
        e1 = str2var['e1']
        e2 = str2var['e2']
        rel = str2var['rel']

        batch_triples = torch.cat((e1, rel, e2), 1)
        triples.append(batch_triples)

        #rel_reverse = str2var['rel_eval']
        e2_multi1 = str2var['e2_multi1'].float()
        e2_multi2 = str2var['e2_multi2'].float()
        pred1 = model.forward(e1, rel)
        pred2 = model.forward(e2, rel)
        pred1, pred2 = pred1.data, pred2.data
        e1, e2 = e1.data, e2.data
        e2_multi1, e2_multi2 = e2_multi1.data, e2_multi2.data
        for i in range(Config.batch_size):
            # these filters contain ALL labels
            filter1 = e2_multi1[i].long()
            filter2 = e2_multi2[i].long()

            num = e1[i, 0].item()
            # save the prediction that is relevant
            target_value1 = pred1[i,e2[i, 0].item()].item()
            target_value2 = pred2[i,e1[i, 0].item()].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[i][filter1] = 0.0
            pred2[i][filter2] = 0.0
            # write base the saved values
            pred1[i][e2[i]] = target_value1
            pred2[i][e1[i]] = target_value2


        # sort and rank
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)

        argsort1 = argsort1.cpu().numpy()
        argsort2 = argsort2.cpu().numpy()
        e1 = e1.cpu().numpy()
        e2 = e2.cpu().numpy()
        for i in range(Config.batch_size):
            # find the rank of the target entities
            rank1 = np.where(argsort1[i]==e2[i, 0])[0][0]
            rank2 = np.where(argsort2[i]==e1[i, 0])[0][0]
            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks.append(rank1+1)
            ranks_left.append(rank1+1)
            ranks.append(rank2+1)
            ranks_right.append(rank2+1)


            # this could be done more elegantly, but here you go
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)

        dev_rank_batcher.state.loss = [0]

    triples = torch.cat(triples, dim=0)
    ent_id_2_uri = vocab['e1']
    rel_id_2_uri = vocab['rel']
    eval_results_file_path = f'saved_models/eval_results_{name}_{model_name}_{literal_representation}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.tsv'
    with open(eval_results_file_path, 'w') as results_out:
        for i in range(triples.size(0)):
            e1 = triples[i, 0].item()
            rel = triples[i, 1].item()
            e2 = triples[i, 2].item()
            #print(ent_id_2_uri.get_word(e1), rel_id_2_uri.get_word(rel), ent_id_2_uri.get_word(e2), ranks_left[i], ranks_right[i])
            results_out.write('\t'.join([ent_id_2_uri.get_word(e1), rel_id_2_uri.get_word(rel), ent_id_2_uri.get_word(e2), str(ranks_left[i]), str(ranks_right[i])]) + '\n')

    for i in range(10):
        log.info('Hits left @{0}: {1}'.format(i+1, np.mean(hits_left[i])))
        log.info('Hits right @{0}: {1}'.format(i+1, np.mean(hits_right[i])))
        log.info('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
    log.info('Mean rank left: {0}', np.mean(ranks_left))
    log.info('Mean rank right: {0}', np.mean(ranks_right))
    log.info('Mean rank: {0}', np.mean(ranks))
    log.info('Mean reciprocal rank left: {0}', np.mean(1./np.array(ranks_left)))
    log.info('Mean reciprocal rank right: {0}', np.mean(1./np.array(ranks_right)))
    log.info('Mean reciprocal rank: {0}', np.mean(1./np.array(ranks)))
