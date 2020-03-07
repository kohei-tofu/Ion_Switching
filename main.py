import util
import argparse
import train
import evaluate
import dataset

def main():

    path_result = './result/'
    util.make_directory(path_result)
    parser = argparse.ArgumentParser(description='PyTorch Kaggle')
    parser.add_argument('--jobtype', '-M', type=str, default='evaluate', help='')
    #parser.add_argument('--model', '-M', type=str, default='', help='')
    args = parser.parse_args()


    import config.setting1 as setting
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



