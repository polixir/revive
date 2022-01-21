import os
import argparse
from revive.utils.license_utils import get_machine_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The file path where the machine information is saved.')
    parser.add_argument('-o', 
                        '--output', 
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "machine_info.json"), 
                        help="The file save path of machine information.")

    args = parser.parse_args()
    get_machine_info(args.output)