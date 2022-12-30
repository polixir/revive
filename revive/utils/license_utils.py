import os
import json
import argparse

from io import StringIO
from wurlitzer import pipes, STDOUT
from pyarmor.pytransform import show_hd_info,pyarmor_init


def get_machine_info(output="./machine_info.json", online=False):
    r"""
    Use pyarmor register a License.

    Args:

    output : A json file with machine information generated using revive sdk. E.g. : "/home/machine_info.json"
   
    """
    assert output.endswith(".json"), f"Machine info should be saved as a json file. -> {output}"

    out = StringIO()
    with pipes(stdout=out, stderr=STDOUT):
        try:
            pyarmor_init()
        except:
            os.system("pyarmor hdinfo")
            pyarmor_init()
        show_hd_info()

    lines = out.getvalue().split("\n")

    hd_info = {
        "harddisk" : [],
        "mac" : [],
        "ip" : [],
    }
    for line in lines:
        if "default harddisk" in line:
            hd_info["harddisk"].append(line[line.index('"')+1:-1])

        if "Default Mac address" in line:
            hd_info["mac"].append(line[line.index('"')+1:-1])

        if "Ip address" in line:
            hd_info["ip"].append(line[line.index('"')+1:-1])
            
    machine_info = {"hd_txt": lines,  "hd_info": hd_info}

    if online:
        with open(output,"w") as f:
            json.dump(machine_info,f)
        with open(output, "r") as f:
            machine_info = f.readlines()[0]
        return machine_info

    with open(output,"w") as f:
        print(f"Svae machine info -> {output}")
        json.dump(machine_info,f)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The file path where the machine information is saved.')
    parser.add_argument('-o', '--output', default="./machine_info.json", help="The file save path of machine information.")

    args = parser.parse_args()
    print(get_machine_info(args.output , online=True))