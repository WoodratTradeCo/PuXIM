import argparse

parser = argparse.ArgumentParser(description='Zero-Shot SBIR')

parser.add_argument('--exp_base', type=str, default='./results')
parser.add_argument('--exp', type=str, default="0", help='result_save_dic')

# --------------------
# DataLoader Options
# --------------------

# Path to 'Sketchy' folder holding Sketch_extended dataset. It should have 2 folders named 'sketch' and 'photo'.
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--data_split', type=float, default=-1.0)
parser.add_argument('--datasetLen', type=int, default=10000)
parser.add_argument('--data_path', type=str, default=r"D:\ZJU\research\datasets\retrieval\Sketchy (low)")
parser.add_argument('--dataset', type=str, default='sketchy_extend',
                    choices=['sketchy_extend', 'tu_berlin', 'Quickdraw'])
parser.add_argument('--test_class', type=str, default='test_class_sketchyam',
                    choices=['test_class_sketchy25', 'test_class_sketchy21', 'test_class_sketchyam',
                             'test_class_tuberlin30', 'Quickdraw'])


# ----------------------
# Training Params
# ----------------------

parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--gpu', type=str, default="0")

# ----------------------
# ViT Prompt Parameters
# ----------------------

parser.add_argument('--match', '-r', type=str, default='mask', choices=['mask', 'non-mask'])
parser.add_argument('--testall', default=True, action='store_true', help='train/test scale')
parser.add_argument('--test_sk', type=int, default=20)
parser.add_argument('--test_im', type=int, default=20)
parser.add_argument('--num_workers', type=int, default=0)


opts = parser.parse_args()


