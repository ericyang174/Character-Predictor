#!/usr/bin/env python
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('fpred')
parser.add_argument('fgold')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()


def load_pred(fname, force_limit=None):
    with open(fname, encoding='utf-8') as f:
        loaded = []
        for line in f:
            line = line[:-1].lower()
            if force_limit is not None:
                line = line[:force_limit]
            loaded.append(line)
        return loaded


pred = load_pred(args.fpred, force_limit=3)
gold = load_pred(args.fgold)

if len(pred) < len(gold):
    pred.extend([''] * (len(gold) - len(pred)))

correct = 0
for i, (p, g) in enumerate(zip(pred, gold)):
    right = g in p
    correct += right
    if args.verbose:
        print('Input {}: {}, {} is {} in {}'.format(i, 'right' if right else 'wrong', g, 'in' if right else 'not in', p))
print('Success rate: {}'.format(correct/len(gold)))