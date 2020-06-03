import json
import argparse
from pandas import *


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--xls-file-path", type=str,
                help="Path to the excel file")
args = vars(ap.parse_args())


def convert_xls_to_json(excel_file_path):
    xls = ExcelFile(excel_file_path)

    df = xls.parse(xls.sheet_names[0])
    with open(f'{xls.sheet_names[0]}.json', "w") as json_file:
        json.dump(df.set_index('id')['opis'].to_dict(), json_file)

    df = xls.parse(xls.sheet_names[1])
    with open(f'{xls.sheet_names[1]}.json', "w") as json_file:
        json.dump(df.set_index('un_broj')['naziv'].to_dict(), json_file)


if __name__ == '__main__':
    convert_xls_to_json(args['xls_file_path'])
