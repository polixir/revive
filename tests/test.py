import os
import pandas as pd

def show_test_results():
    r"""
    Run test scipts in 'test.sh'.
   
    """
    test_scipt_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "test_result.out")
    with open(test_scipt_path, "r") as f:
        lines = f.readlines()

    warning_index = []
    warning_flag = []
    warning_line = []
    for line_index, line in enumerate(lines):
        # check warning
        for e in ["warning","WARNING","Warning"]:
            if e in line:
                warning_index.append(line_index)
                warning_flag.append(e)
                warning_line.append(line)
    warning_res = pd.DataFrame([warning_index,warning_flag,warning_line]).T
    warning_res.columns = ['Line index','Warning Flag', 'Line text']
    if warning_res.shape[0] == 0:
        print("Not found Warning.")
    else:
        print(f"Found Warning in -> {test_scipt_path}")
        print(warning_res)


    error_index = []
    error_flag = []
    error_line = []
    for line_index, line in enumerate(lines):
        # check error
        for e in ["error","ERROR","Error"]:
            if e in line:
                error_index.append(line_index)
                error_flag.append(e)
                error_line.append(line)
    error_res = pd.DataFrame([error_index,error_flag,error_line]).T
    error_res.columns = ['Line index','Error Flag', 'Line text']
    if error_res.shape[0] == 0:
        print("Not found Error.")
    else:
        print(f"Found Error in -> {test_scipt_path}")
        print(error_res)


if __name__ == "__main__":
    show_test_results()