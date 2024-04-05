import os
import errno
from pathlib import Path
import pickle

import numpy as np

from collections import defaultdict

import argparse

DATA_PATH = "data"

parser = argparse.ArgumentParser(
    description="Knowledge Graph Completion"
)

parser.add_argument(
    '--directory', type=str, default='GO0404'
)

args = parser.parse_args()

def prepare_dataset(path, name): # /src_data/GO0608, GO0608
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    files = ['train', 'valid', 'test']
    entities, relations = set(), set()
    for f in files:
        with open(path+'/'+f+'.pickle', 'rb') as p:
            file = pickle.load(p)
            for line in file:
                entities.add(line[0])
                entities.add(line[2])
                relations.add(line[1])

    print("{} entities and {} relations".format(len(entities), len(relations)))
    n_relations = len(relations)
    n_entities = len(entities)

    print("creating filtering lists")

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs in examples:
            to_skip['lhs'][(rhs, rel + n_relations)].add(lhs)
            to_skip['rhs'][(lhs, rel)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(Path(DATA_PATH) / name / 'train.pickle', 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(DATA_PATH) / name / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()


if __name__ == "__main__":
    datasets = ['GO0404', 'GOA0404']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_dataset(
                os.path.join(
                    'data', d
                ),
                d
            )
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise