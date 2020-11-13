from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/data/dataset/test-split/dataset_coco.json',
                       image_folder='/data/dataset/coco/Images/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='../data_outdoor_full_coco_blurred/min1e-10tomax6/',
                       max_len=50,
                      needOutdoor= True,
                      toBlur=True,
                      minSigma=1e-10,
                      maxSigma=6)
