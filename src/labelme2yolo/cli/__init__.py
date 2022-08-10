# SPDX-FileCopyrightText: 2022-present Wang Xin <xinwang614@gmail.com>
#
# SPDX-License-Identifier: MIT
import argparse

from labelme2yolo.l2y import Labelme2YOLO


def run():
    parser = argparse.ArgumentParser("labelme2yolo")
    parser.add_argument(
        "--json_dir", type=str, help="Please input the path of the labelme json files."
    )
    parser.add_argument(
        "--val_size",
        type=float,
        nargs="?",
        default=None,
        help="Please input the validation dataset size, for example 0.1 ",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        nargs="?",
        default=None,
        help="Please input the validation dataset size, for example 0.1 ",
    )
    parser.add_argument(
        "--json_name",
        type=str,
        nargs="?",
        default=None,
        help="If you put json name, it would convert only one json file to YOLO.",
    )
    args = parser.parse_args()

    if not args.json_dir:
        parser.print_help()
        return 0

    convertor = Labelme2YOLO(args.json_dir)

    if args.json_name is None:
        convertor.convert(val_size=args.val_size, test_size=args.test_size)
    else:
        convertor.convert_one(args.json_name)

    return 0
