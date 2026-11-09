import argparse
from utils.seed import seed_everything

def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['download','train-pretrain','train-linear','train-supervised','evaluate','demo']); p.add_argument('--epochs',type=int,default=1); p.add_argument('--limit',type=int,default=256); args=p.parse_args(); seed_everything()
    if args.command=='download':
        from datasets.cifar10 import download_cifar10; print(download_cifar10())
    elif args.command=='train-pretrain':
        from training.pretrain import train; train(args.epochs,limit=args.limit)
    elif args.command=='train-linear':
        from training.linear_eval import train; train(args.epochs,limit=args.limit)
    elif args.command=='train-supervised':
        from training.supervised import train; train(args.epochs,limit=args.limit)
    elif args.command=='evaluate':
        from evaluation.evaluate import evaluate_linear,evaluate_supervised,save_report
        report={'linear': evaluate_linear(limit=args.limit), 'supervised': evaluate_supervised(limit=args.limit)}; save_report(report)
        print(report)
    else:
        from app.demo import launch; launch()
if __name__=='__main__': main()
