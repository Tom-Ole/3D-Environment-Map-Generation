import argparse
import os

def getNames(args):
    dir = args.directory

    return [x for x in os.listdir(dir)]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get a list of all folder names from a given dircetory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--directory", default=".",
        help="",
    )

    names = getNames(parser.parse_args())

    [print(f"{x}") for x in names]