import json
import argparse
from trainer import train
from evaluator import evaluate

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    if args.init_epoch:
        param['init_epoch'] = args.init_epoch
    if args.epochs:
        param['epochs'] = args.epochs
    if args.initial_checkpoint:
        param['initial_checkpoint'] = args.initial_checkpoint
    if args.init_cls:
        param['init_cls'] = args.init_cls
    if args.increment:
        param['increment'] = args.increment
    if args.total_sessions:
        param['total_sessions'] = args.total_sessions
    if args.init_lr:
        param['init_lr'] = args.init_lr
    if args.lrate:
        param['lrate'] = args.lrate
    if args.batch_size:
        param['batch_size'] = args.batch_size
    if args.rank:
        param['rank'] = args.rank
    if args.lamb:
        param['lamb'] = args.lamb
    if args.lame:
        param['lame'] = args.lame
    if args.alpha_lora:
        param['alpha_lora'] = args.alpha_lora
    param['model'] = args.model
    param['exp'] = args.exp
    param['finetune'] = args.finetune
    param['use_gpm'] = args.use_gpm
    param['use_dualgpm'] = args.use_dualgpm
    param['alpha_gpm'] = args.alpha_gpm
    param['tasktest'] = args.tasktest
    if args.seed:
        param['seed'] = args.seed
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    if args['test']:
        evaluate(args)
    else:
        train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--init_epoch', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--model', type=str, default='MONet_S')
    parser.add_argument('--exp', type=str, default='test')
    parser.add_argument("--finetune", nargs="+", type=str, help="finetune choices")
    parser.add_argument('--init_lr', type=float, default=None)
    parser.add_argument('--lrate', type=float, default=None)
    parser.add_argument('--alpha_lora', type=float, default=None)
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--use_gpm', action="store_true", default=False, help="Use Gradient Projection Memory or not")
    parser.add_argument('--use_dualgpm', action="store_true", default=False, help="Use Dual Gradient Projection Memory or not")
    parser.add_argument('--lamb', type=float, default=None)
    parser.add_argument('--lame', type=float, default=None)
    parser.add_argument('--alpha_gpm', type=float, default=1)
    parser.add_argument('--tasktest', type=int, default=1)
    parser.add_argument('--test', action="store_true", default=False, help="test or not")
    parser.add_argument('--initial_checkpoint', type=str, default=None)
    parser.add_argument('--seed', nargs="+", type=int, default=None)
    parser.add_argument('--init_cls', type=int, default=None)
    parser.add_argument('--increment', type=int, default=None)
    parser.add_argument('--total_sessions', type=int, default=None)

    # # optim
    # parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'])
    return parser


if __name__ == '__main__':
    main()