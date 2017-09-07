import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import argparse
import codecs
import json
import os

import numpy as np
from ch2ch.char_rnn_model import *
from ch2ch.train import load_vocab

g_params=dict()
def init(model_dir='model'):
    parser = argparse.ArgumentParser()
    
    # Parameters for using saved best models.
    parser.add_argument('--init_dir', type=str, default=model_dir,
                        help='continue from the outputs in the given directory')

    # Parameters for picking which model to use. 
    parser.add_argument('--model_path', type=str, default=os.path.join(model_dir, 'best_model/model-2418'),
                        help='path to the model file like output/best_model/model-40')

    # Parameters for sampling.
    parser.add_argument('--temperature', type=float,
                        default=1.0,
                        help=('Temperature for sampling from softmax: '
                              'higher temperature, more random; '
                              'lower temperature, more greedy.'))

    parser.add_argument('--max_prob', dest='max_prob', action='store_true',
                        help='always pick the most probable next character in sampling')

    parser.set_defaults(max_prob=False)

    parser.add_argument('--start_text', type=str,
                        default='Здравствуйте нет зачисления зарплаты по реестру 9 от 31.03.17 года, реестр висит со вчерашнего дня?',
                        help='the text to start with')

    parser.add_argument('--length', type=int,
                        default=150,
                        help='length of sampled sequence')

    parser.add_argument('--seed', type=int,
                        default=-1,
                        help=('seed for sampling to replicate results, '
                              'an integer between 0 and 4294967295.'))

    # Parameters for evaluation (computing perplexity of given text).
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='compute the perplexity of given text')
    parser.set_defaults(evaluate=False)
    parser.add_argument('--example_text', type=str,
                        default='Здравствуйте нет зачисления зарплаты по реестру 9 от 31.03.17 года, реестр висит со вчерашнего дня?',
                        help='compute the perplexity of given example text.')


    # Parameters for debugging.
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='show debug information')
    parser.set_defaults(debug=False)

    args = parser.parse_args([])


    # Prepare parameters.
    with open(os.path.join(args.init_dir, 'result.json'), 'r') as f:
        result = json.load(f)
    params = result['params']


    if args.model_path:    
        best_model = args.model_path
    else:
        best_model = result['best_model']

    best_valid_ppl = result['best_valid_ppl']
    if 'encoding' in result:
        args.encoding = result['encoding']
    else:
        args.encoding = 'utf-8'
    args.vocab_file = os.path.join(args.init_dir, 'vocab.json')
    vocab_index_dict, index_vocab_dict, vocab_size = load_vocab(args.vocab_file, args.encoding)


    # Create graphs
    logging.info('Creating graph')
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('evaluation'):
            test_model = CharRNN(is_training=False, use_batch=False, **params)
            saver = tf.train.Saver(name='checkpoint_saver')

    if args.seed >= 0:
        np.random.seed(args.seed)
    g_params['graph'] = graph
    g_params['saver'] = saver
    g_params['best_model'] = best_model
    g_params['test_model'] = test_model
    g_params['length'] = args.length
    g_params['vocab_index_dict'] = vocab_index_dict
    g_params['index_vocab_dict'] = index_vocab_dict
    g_params['temperature'] = args.temperature
    g_params['max_prob'] = args.max_prob



def send(msg='Здравствуйте.'):
    # Sampling a sequence 
    with tf.Session(graph=g_params['graph']) as session:
        g_params['saver'].restore(session, g_params['best_model'])
        sample = g_params['test_model'].sample_seq(session, g_params['length'], msg,
                                        g_params['vocab_index_dict'], g_params['index_vocab_dict'],
                                        temperature=g_params['temperature'],
                                        max_prob=g_params['max_prob'])
        sample = sample.split('\n')[1]
    return sample

if __name__ == '__main__':
    init()
