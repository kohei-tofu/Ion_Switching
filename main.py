import util
import argparse
import train

def main():

    path_result = './result/'
    util.make_directory(path_result)
    job_type = 'train'
    #parser = argparse.ArgumentParser(description='PyTorch Kaggle')
    #parser.add_argument('--model', '-M', type=str, default='', help='')
    #args = parser.parse_args()


    import config.setting1 as setting
    cfg = setting.Config()

    if job_type == 'train':
        train.main(cfg)

    if jobtype == 'evaluate':
        evaluate.main(cfg)

if __name__ == '__main__':

    print('start')

    main()
    #main1()

    print('end')



