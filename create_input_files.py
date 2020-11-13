from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/data/dataset/test-split/dataset_flickr8k.json',
                       image_folder='/data/dataset/flickr-8k/Flickr_Data/Flickr_Data/Images/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='../output',
                       max_len=50)
