# SPDX-FileCopyrightText: 2022-present Wang Xin <xinwang614@gmail.com>
#
# SPDX-License-Identifier: MIT
"""
main
"""
import sys

if __name__ == "__main__":
    from .cli import run

    sys.exit(run())
