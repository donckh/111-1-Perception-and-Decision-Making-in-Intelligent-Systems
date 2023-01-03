This readme assuming the dataset are well prepare or you need to modify data_generate.py, create_json.py according to my report mentioned:
Task 1:
prepare training data:
1. We need to prepare the dataset first, implement data_generate.py:
python data_generator.py --output [Output folder] --dataset_folder [--dataset_folder]
2. Modify load.py from hw1 as report mentioned.
3. run create_json.py to prepare the json file for training.

Training:
1. Modify and run the train.py as mentioned in report, it will start from epoch 20, make sure you have epoch_20.pth at the correct DIR.
python train.py --gpus 0 --cfg [Model config DIR]

Validate:
2. Change number of class at eval_multipro.py in main()
cfg.DATASET.num_class = 101

3. execute eval_multipro.py
python3 eval_multipro.py --cfg [Model config DIR] --gpus 0

Task 2:
1. Modify directory at 3d_semantic_map.py to correct directory which you save the images to be reconstruct.

path in main() function:
path = 'semantic/images/directory'

path for read image at o3d_img_to_3d function:
source_color = o3d.io.read_image("semantic/images/directory/{}.png".format(image_name))
source_depth = o3d.io.read_image("depth/images/directory/{}.png".format(image_name))

2. Execute the 3d_semantic_map.py and check the reconstruction finally.
