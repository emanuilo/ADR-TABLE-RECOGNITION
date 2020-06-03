import os
import re
import cv2
import json
import argparse
import pytesseract
from darkflow.net.build import TFNet
from pandas import *


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                help="path to input image")
ap.add_argument("-e", "--convert-xls-to-json", type=str,
                help="Convert excel file to json")
ap.add_argument("-id", "--image-dir", type=str,
                help="path to input directory")
ap.add_argument("-gtd", "--ground-truth-dir", type=str,
                help="path to ground truth directory")
ap.add_argument("-d", "--demo", action="store_true",
                help="Run demo")
ap.add_argument("-t", "--threshold", type=float, default=0.2,
                help="minimum confidence for predicted bounding boxes")
args = vars(ap.parse_args())

OUT_DIR = 'out/'


class ImagePreprocessor:

    @staticmethod
    def adaptive_preproc(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 11)
        return img


class ImageUtil:

    @staticmethod
    def crop_number_image(img, predictions, expand=True):
        cropped_images = []
        for prediction in predictions:
            xtop = prediction.get('topleft').get('x')
            ytop = prediction.get('topleft').get('y')
            xbottom = prediction.get('bottomright').get('x')
            ybottom = prediction.get('bottomright').get('y')

            if expand:
                xtop, ytop, xbottom, ybottom = ImageUtil.expand_number_box(xtop, ytop, xbottom, ybottom)

            cropped_images.append(img[ytop:ybottom, xtop:xbottom])
        return cropped_images

    @staticmethod
    def crop_image(img, predictions, expand=True):
        predictions.sort(key=lambda x: x.get('confidence'), reverse=True)

        if predictions:
            xtop = predictions[0].get('topleft').get('x')
            ytop = predictions[0].get('topleft').get('y')
            xbottom = predictions[0].get('bottomright').get('x')
            ybottom = predictions[0].get('bottomright').get('y')

            if expand:
                xtop, ytop, xbottom, ybottom = ImageUtil.expand_box(xtop, ytop, xbottom, ybottom)

            crop = img[ytop:ybottom, xtop:xbottom]
            return crop
        return img

    @staticmethod
    def expand_number_box(xtop, ytop, xbottom, ybottom, percentage=0.1, percentage2=0.15):
        xtop = int(xtop - (xbottom - xtop) * percentage)
        ytop = int(ytop - (ybottom - ytop) * percentage2)
        xbottom = int(xbottom + (xbottom - xtop) * percentage2)
        ybottom = int(ybottom + (ybottom - ytop) * percentage2)
        return xtop, ytop, xbottom, ybottom

    @staticmethod
    def expand_box(xtop, ytop, xbottom, ybottom, percentage=0.2):
        xtop = max((int(xtop - (xbottom - xtop) * percentage), 0))
        ytop = max((int(ytop - (ybottom - ytop) * percentage), 0))
        xbottom = int(xbottom + (xbottom - xtop) * percentage)
        ybottom = int(ybottom + (ybottom - ytop) * percentage)
        return xtop, ytop, xbottom, ybottom

    @staticmethod
    def no_extension_file_name(file_path):
        return os.path.splitext(os.path.basename(file_path))[0]


class AdrTableDetector:

    def __init__(self, threshold):
        options = {"model": "cfg/yolov2-large-ann.cfg", "load": 1000, "threshold": threshold,
                   "labels": "labels/adr_table_label.txt"}
        self.tfnet = TFNet(options)
        self.tfnet.load_from_ckpt()

    def detect_adr_table(self, img):
        results = self.tfnet.return_predict(img)
        cropped_image = ImageUtil.crop_image(img, results)
        cropped_image = cv2.medianBlur(cropped_image, 1)
        return cropped_image


class NumberDetector:

    def __init__(self, threshold):
        options = {"model": "cfg/yolov2-numbers.cfg", "load": 1250, "threshold": threshold,
                   "labels": "labels/number_label.txt"}
        self.tfnet = TFNet(options)
        self.tfnet.load_from_ckpt()
        self.tesseract_conf = '--oem 1 --psm 10 outputbase digits'

    def detect_numbers(self, img, file_name):
        results = self.tfnet.return_predict(img)
        first_row_results, second_row_results = self.split_rows(results)
        first_row_images = ImageUtil.crop_number_image(img, first_row_results)
        second_row_images = ImageUtil.crop_number_image(img, second_row_results)

        first_row_numbers = self.get_numbers(first_row_images)
        # print(f'First row: {first_row_numbers}')
        second_row_numbers = self.get_numbers(second_row_images)
        # print(f'Second row: {second_row_numbers}')
        return first_row_numbers, second_row_numbers

    def get_numbers(self, images):
        numbers = []
        for cropped_image in images:
            preproc_image = ImagePreprocessor.adaptive_preproc(cropped_image)
            number_result = pytesseract.image_to_string(preproc_image, config=self.tesseract_conf)
            numbers.append(number_result)
        return ''.join(map(str, numbers))

    def split_rows(self, predictions):
        predictions.sort(key=lambda x: x.get('topleft').get('y'), reverse=False)
        numbers_in_first_row = self.count_first_row_numbers(predictions)

        first_row_predictions = predictions[:numbers_in_first_row]
        first_row_predictions.sort(key=lambda x: x.get('topleft').get('x'), reverse=False)

        second_row_predictions = predictions[numbers_in_first_row:]
        second_row_predictions.sort(key=lambda x: x.get('topleft').get('x'), reverse=False)

        return first_row_predictions, second_row_predictions

    def count_first_row_numbers(self, predictions):
        try:
            first_y = predictions[0].get('topleft').get('y')
            second_y = predictions[1].get('topleft').get('y')
            ref_difference = second_y - first_y

            MAGIC_NUMBER = 3
            for i in range(1, len(predictions)):
                temp_y2 = predictions[i + 1].get('topleft').get('y')
                temp_y1 = predictions[i].get('topleft').get('y')
                if temp_y2 - temp_y1 > ref_difference * MAGIC_NUMBER:
                    return i + 1
        except IndexError:
            print('Not enough data.')
            return 0


class NumberToSubstanceConv:

    def __init__(self, substances_file_path, dangers_file_path):
        with open(substances_file_path, "r") as json_file:
            self.substances_dict = json.load(json_file)

        with open(dangers_file_path, "r") as json_file:
            self.dangers_dict = json.load(json_file)

    @staticmethod
    def convert_xls_to_json(excel_file_path):
        xls = ExcelFile(excel_file_path)

        df = xls.parse(xls.sheet_names[0])
        with open(f'{xls.sheet_names[0]}.json', "w") as json_file:
            json.dump(df.set_index('id')['opis'].to_dict(), json_file)

        df = xls.parse(xls.sheet_names[1])
        with open(f'{xls.sheet_names[1]}.json', "w") as json_file:
            json.dump(df.set_index('un_broj')['naziv'].to_dict(), json_file)

    def get_substance_name(self, substance_id):
        try:
            substance_name = self.substances_dict[substance_id]
            return substance_name
        except KeyError:
            print("Key Error!")
            return ''

    def get_danger_name(self, danger_id):
        try:
            danger_name = self.dangers_dict[danger_id]
            return danger_name
        except KeyError:
            print("Key Error!")
            return ''


if __name__ == '__main__':
    if args['convert_xls_to_json']:
        converter = NumberToSubstanceConv('substances_files/Materija.json', 'substances_files/IdentifikacijaOpasnosti.json')
    elif args['demo']:
        thresh = args['threshold']
        adr_table_detector = AdrTableDetector(args['threshold'])
        number_detector = NumberDetector(args['threshold'])

        ground_truth_dir = args['ground_truth_dir']
        image_dir = args['image_dir']
        correct_cnt = 0
        total_cnt = 0

        for image_path in os.listdir(image_dir):
            print(f'------------{image_path}------------')
            if image_path.endswith('.jpg'):
                base_name = os.path.splitext(image_path)[0]
                img = cv2.imread(os.path.join(image_dir, image_path))
                adr_table = adr_table_detector.detect_adr_table(img)
                # adr_table = cv2.medianBlur(adr_table, 1)
                first_row, second_row = number_detector.detect_numbers(
                    adr_table,
                    ImageUtil.no_extension_file_name(image_path)
                )
                first_row = re.sub('[^0-9]', '', first_row)
                second_row = re.sub('[^0-9]', '', second_row)
                # print(converter.get_danger_name(first_row))
                # print(converter.get_substance_name(second_row))
                with open(f'{ground_truth_dir}/{base_name}.json') as f:
                    ground_truth = json.load(f)

                if first_row == ground_truth['first_row'] and second_row == ground_truth['second_row']:
                    correct_cnt += 1
                total_cnt += 1

        accuracy = correct_cnt / total_cnt
        print(f'Correct: {correct_cnt}')
        print(f'Total: {total_cnt}')
        print(f'Accuracy: {accuracy}%')
    else:
        thresh = args['threshold']
        adr_table_detector = AdrTableDetector(args['threshold'])
        number_detector = NumberDetector(args['threshold'])

        if args['image_dir']:
            for image_path in os.listdir(args['image_dir']):
                img = cv2.imread(os.path.join(args['image_dir'], image_path))
                adr_table = adr_table_detector.detect_adr_table(img)
                number_detector.detect_numbers(adr_table, ImageUtil.no_extension_file_name(image_path))

        elif args['image']:
            img = cv2.imread(args['image'])
            adr_table = adr_table_detector.detect_adr_table(img)
            number_detector.detect_numbers(adr_table, ImageUtil.no_extension_file_name(args['image']))
