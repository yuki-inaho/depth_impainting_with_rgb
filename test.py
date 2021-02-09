
import click
import cv2
from pathlib import Path
from scripts.impainting import ImpaintingModule
from scripts.image import colorize_uint_image
import pickle


@click.command()
@click.option("--architecture", "-a", default="DM-LRN", help="Archtecture type in [LRN, DM-LRN]")
@click.option("--model-config-file", "-m", default="cfg/DM-LRN_efficientnet-b4.yaml", help="Location of config file")
@click.option("--weight-file", "-w", required=True, help="Location of weight file")
@click.option("--input-data-dir", "-i", default="./data")
@click.option("--image-height", "-ih", default=1920)
@click.option("--image-width", "-iw", default=1080)
def main(architecture, model_config_file, weight_file, input_data_dir, image_height, image_width):
    input_dir_path = Path(input_data_dir)
    rgb_dir_path = input_dir_path.joinpath("rgb")
    depth_dir_path = input_dir_path.joinpath("depth")

    #rgbd_depth_obj_list: List[RGBDImage] = extract_rgbd_result(rgb_dir_path, depth_dir_path)
    #with open('test.pickle', 'wb') as f:
    #    pickle.dump(rgbd_depth_obj_list, f)
    with open('test.pickle', 'rb') as f:
        rgb_depth_obj_list = pickle.load(f)

    '''
    depth_mean_list = []
    depth_std_list = []
    depth_min = 10000
    depth_max = 0
    for rgb_depth_obj in rgb_depth_obj_list:
        _depth_max = rgb_depth_obj.depth.max()
        _depth_min = rgb_depth_obj.depth.min()
        depth_max = _depth_max if _depth_max > depth_max else depth_max
        depth_min = _depth_min if _depth_min < depth_min else depth_min
        _depth_mean = rgb_depth_obj.depth.mean()
        _depth_std = rgb_depth_obj.depth.std()
        depth_mean_list.append(_depth_mean)
        depth_std_list.append(_depth_std)
    '''

    depth_mean = 50.52094613381737
    depth_std = 51.5248996299203
    impainter: ImpaintingModule = ImpaintingModule(architecture, model_config_file, weight_file, image_width, image_height)
    impainter.set_depth_mean_and_std(depth_mean, depth_std)
    #predicted_depth = impainter.inference(rgb_depth_obj_list[0])
    #colorized_depth = colorize_uint_image(predicted_depth)
    #cv2.imwrite("colorized.png", colorized_depth)
    #cv2.waitKey(10)


if __name__ == "__main__":
    main()
