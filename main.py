import util
import argparse
import train
import evaluate
import dataset

def main():

    parser = argparse.ArgumentParser(description='PyTorch Kaggle')
    parser.add_argument('--jobtype', '-M', type=str, default='evaluate', help='what are you going to do on this function')
    parser.add_argument('--setting', '-S', type=str, default='setting1', help='which setting file are you going to use for training.')
    #parser.add_argument('--model', '-M', type=str, default='', help='')
    args = parser.parse_args()


    #import config.setting1 as setting
    from config import eval(args.setting) as setting
    cfg = setting.Config()

    if args.jobtype == 'preprocess':
        dataset.main(cfg)

    if args.jobtype == 'train':
        train.main(cfg)

    if args.jobtype == 'evaluate':
        evaluate.main(cfg)

if __name__ == '__main__':

    print('start')

    main()

    print('end')



