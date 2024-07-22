import csv
import random
import argparse
import numpy as np
import os
import tensorflow.compat.v1 as tf

MIN = '[MIN'
MAX = '[MAX'
MED = '[MED'
FIRST = '[FIRST'
LAST = '[LAST'
SUM_MOD = '[SM'
END = ']'

OPERATORS = [MIN, MAX, MED, SUM_MOD]  # , FIRST, LAST]
VALUES = range(10)

VALUE_P = 0.25


def generate_tree(depth, max_depth, max_args):
  """Generate tree-like equations.

  Args:
    depth: current depth of the node, int.
    max_depth: maximum depth of the tree, int.
    max_args: maximum number of arguments per operator, int.

  Returns:
    The root node of a tree structure.
  """
  if depth < max_depth:
    r = random.random()
  else:
    r = 1

  if r > VALUE_P:
    value = random.choice(VALUES)
    return value, 1
  else:
    length = 2
    num_values = random.randint(2, max_args)
    values = []
    for _ in range(num_values):
      sub_t, sub_l = generate_tree(depth + 1, max_depth, max_args)
      values.append(sub_t)
      length += sub_l

    op = random.choice(OPERATORS)
    t = (op, values[0])
    for value in values[1:]:
      t = (t, value)
    t = (t, END)
  return t, length


def to_string(t, parens=True):
  if isinstance(t, str):
    return t
  elif isinstance(t, int):
    return str(t)
  else:
    if parens:
      return '( ' + to_string(t[0]) + ' ' + to_string(t[1]) + ' )'


def to_value(t):
  """Compute the output of equation t.

  Args:
    t: a tree structure that represents equation t, list.

  Returns:
    The result of equation t, int.
  """
  if not isinstance(t, tuple):
    return t
  l = to_value(t[0])
  r = to_value(t[1])
  if l in OPERATORS:  # Create an unsaturated function.
    return (l, [r])
  elif r == END:  # l must be an unsaturated function.
    if l[0] == MIN:
      return min(l[1])
    elif l[0] == MAX:
      return max(l[1])
    elif l[0] == FIRST:
      return l[1][0]
    elif l[0] == LAST:
      return l[1][-1]
    elif l[0] == MED:
      return int(np.median(l[1]))
    elif l[0] == SUM_MOD:
      return np.sum(l[1]) % 10
  elif isinstance(l, tuple):
    # We've hit an unsaturated function and an argument.
    return (l[0], l[1] + [r])


def write_to_file(data, fp):
  """Write to file output."""
  print('Writing {} samples to {}'.format(len(data), fp + '.tsv'))
  with tf.io.gfile.GFile(fp + '.tsv', 'w+') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['Source', 'Target'])
    writer.writerows(data)


def main(args):
  print('Start dataset construction')

  data = set()
  num_samples = args.num_train_samples + args.num_test_samples + args.num_valid_samples
  while len(data) < num_samples:
    tree, length = generate_tree(1, args.max_depth, args.max_args)
    if length > args.min_length and length < args.max_length:
      data.add(tree)
      if len(data) % 1000 == 0:
        print('Processed {}'.format(len(data)))
  train = []
  for example in data:
    train.append([to_string(example), to_value(example)])

  print('Finished running dataset construction')

  val = train[args.num_train_samples:]
  test = val[args.num_valid_samples:]
  val = val[:args.num_valid_samples]
  train = train[:args.num_train_samples]

  print('Dataset size: {}/{}/{}'.format(len(train), len(val), len(test)))

  script_dir = os.path.dirname(os.path.abspath(__file__))
  output_dir = '{}_{}-{}'.format(args.task, args.min_length, args.max_length)
  output_dir = os.path.join(script_dir, output_dir)
    
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  write_to_file(train, os.path.join(output_dir, 'train'))
  write_to_file(val, os.path.join(output_dir, 'val'))
  write_to_file(test, os.path.join(output_dir, 'test'))
  print('Finished writing all to file')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate ListOps dataset.')
  parser.add_argument('--task', default='basic', help='Name of task to create.')
  parser.add_argument('--num_train_samples', type=int, default=96000, help='Number of train samples.')
  parser.add_argument('--num_valid_samples', type=int, default=2000, help='Number of valid samples.')
  parser.add_argument('--num_test_samples', type=int, default=2000, help='Number of test samples.')
  parser.add_argument('--max_depth', type=int, default=10, help='Maximum tree depth of training sequences.')
  parser.add_argument('--max_args', type=int, default=10, help='Maximum number of arguments per operator in training sequences.')
  parser.add_argument('--max_length', type=int, default=2000, help='Maximum length per sequence in training sequences.')
  parser.add_argument('--min_length', type=int, default=500, help='Minimum length per sequence in training sequences.')

  args = parser.parse_args()
  main(args)