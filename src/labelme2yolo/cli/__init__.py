# SPDX-FileCopyrightText: 2022-present Wang Xin <xinwang614@gmail.com>
#
# SPDX-License-Identifier: MIT
"""
cli init
"""
import argparse
from labelme2yolo.__about__ import __version__
from labelme2yolo.l2y import Labelme2YOLO


def run():
    """
    run cli
    """
    parser = argparse.ArgumentParser("labelme2yolo")
    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s " + __version__
    )
    parser.add_argument(
        "--json_dir", type=str, help="Please input the path of the labelme json files."
    )
    parser.add_argument(
        "--val_size",
        type=float,
        nargs="?",
        default=0.2,
        help="Please input the validation dataset size, for example 0.2.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        nargs="?",
        default=0.0,
        help="Please input the test dataset size, for example 0.1.",
    )
    parser.add_argument(
        "--json_name",
        type=str,
        nargs="?",
        default=None,
        help="If you put json name, it would convert only one json file to YOLO.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="polygon",
        help='The default output format for labelme2yolo is "polygon".'
        ' However, you can choose to output in bbox format by specifying the "bbox" option.',
    )
    parser.add_argument(
        "--label_list",
        type=str,
        nargs="+",
        default=None,
        help="The ordered label list, for example --label_list cat dog",
        required=False,
    )

    args = parser.parse_args()

    if not args.json_dir:
        parser.print_help()
        return 0

    convertor = Labelme2YOLO(args.json_dir, args.output_format, args.label_list)

    if args.json_name is None:
        convertor.convert(val_size=args.val_size, test_size=args.test_size)
    else:
        convertor.convert_one(args.json_name)

    return 0
