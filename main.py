import os
import re
import cv2
import json
import logging
import argparse
import pytesseract
from darkflow.net.build import TFNet
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Image, SimpleDocTemplate, Table

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                help="path to input image")
ap.add_argument("-e", "--convert-xls-to-json", type=str,
                help="Convert excel file to json")
ap.add_argument("-vt", "--validation-test", action='store_true',
                help="Command for a validation test. It generates PDF report. "
                     "It it is compatible with --image-dir and --ground-truth-dir.")
ap.add_argument("-id", "--image-dir", type=str, default='test_images',
                help="path to input directory")
ap.add_argument("-gtd", "--ground-truth-dir", type=str, default='test_images/ground_truth',
                help="path to ground truth directory")
ap.add_argument("-t", "--threshold", type=float, default=0.2,
                help="minimum confidence for predicted bounding boxes")
args = vars(ap.parse_args())

OUT_DIR = 'out/'
TEMP_IMAGE_DIR = os.path.join(OUT_DIR, 'temp')
INFERENCE_RESULTS_FILE_NAME = 'results.json'


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
        first_row_results, second_row_results = self.__split_rows(results)
        first_row_images = ImageUtil.crop_number_image(img, first_row_results)
        second_row_images = ImageUtil.crop_number_image(img, second_row_results)

        return first_row_images, second_row_images

    def get_rows(self, first_row_images, second_row_images):
        first_row_numbers = self.__get_numbers(first_row_images)
        second_row_numbers = self.__get_numbers(second_row_images)

        first_row_numbers = re.sub('[^0-9]', '', first_row_numbers)
        second_row_numbers = re.sub('[^0-9]', '', second_row_numbers)

        return first_row_numbers, second_row_numbers

    def __get_numbers(self, images):
        numbers = []
        for cropped_image in images:
            preproc_image = ImagePreprocessor.adaptive_preproc(cropped_image)
            number_result = pytesseract.image_to_string(preproc_image, config=self.tesseract_conf)
            numbers.append(number_result)
        return ''.join(map(str, numbers))

    def __split_rows(self, predictions):
        predictions.sort(key=lambda x: x.get('topleft').get('y'), reverse=False)
        numbers_in_first_row = self.__count_first_row_numbers(predictions)

        first_row_predictions = predictions[:numbers_in_first_row]
        first_row_predictions.sort(key=lambda x: x.get('topleft').get('x'), reverse=False)

        second_row_predictions = predictions[numbers_in_first_row:]
        second_row_predictions.sort(key=lambda x: x.get('topleft').get('x'), reverse=False)

        return first_row_predictions, second_row_predictions

    def __count_first_row_numbers(self, predictions):
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
            # print('Not enough data.')
            return 0


class NumberToSubstanceConv:

    def __init__(self, substances_file_path, dangers_file_path):
        with open(substances_file_path, "r") as json_file:
            self.substances_dict = json.load(json_file)

        with open(dangers_file_path, "r") as json_file:
            self.dangers_dict = json.load(json_file)

    def get_substance_name(self, substance_id):
        try:
            substance_name = self.substances_dict[substance_id]
            return substance_name
        except KeyError:
            # print("Key Error!")
            return ''

    def get_danger_name(self, danger_id):
        try:
            danger_name = self.dangers_dict[danger_id]
            return danger_name
        except KeyError:
            # print("Key Error!")
            return ''


class ReportGenerator:

    def __init__(self):
        self.story = []
        self.styles = getSampleStyleSheet()
        # alignment - TA_LEFT, TA_CENTER, TA_CENTRE, TA_RIGHT, TA_JUSTIFY => 0, 1, 2, 3, 4
        self.normal_style = ParagraphStyle(
            'normal', alignment=0, parent=self.styles['Normal'])
        self.heading1_style = ParagraphStyle(
            'heading1', alignment=0, parent=self.styles['Heading1'], spaceBefore=20)
        if not os.path.exists(TEMP_IMAGE_DIR):
            os.makedirs(TEMP_IMAGE_DIR)
        self.temp_img_cnt = 0

    def add_heading(self, text):
        self.story.append(Paragraph(text, self.heading1_style))

    def add_text(self, text):
        self.story.append(Paragraph(text, self.normal_style))

    def add_image(self, img, height=150):
        resized_img = self.__resize_image(img, height)
        temp_img_name = self.__get_temp_img_name()
        cv2.imwrite(temp_img_name, resized_img)
        self.story.append(Image(temp_img_name, hAlign='LEFT'))

    def add_number_images(self, images, height=70):
        table_row = []

        for img in images:
            preproc_img = ImagePreprocessor.adaptive_preproc(img)
            resized_img = self.__resize_image(preproc_img, height)
            temp_img_name = self.__get_temp_img_name()
            cv2.imwrite(temp_img_name, resized_img)
            table_row.append(Image(temp_img_name, hAlign='LEFT'))

        if len(table_row) > 0:
            self.story.append(Table([table_row], hAlign='LEFT'))

    def generate_report(self, output_filename):
        doc = SimpleDocTemplate(output_filename + '.pdf', pagesize=letter)
        doc.build(self.story)

        for temp_img in os.listdir(TEMP_IMAGE_DIR):
            os.remove(os.path.join(TEMP_IMAGE_DIR, temp_img))
        os.removedirs(TEMP_IMAGE_DIR)

    def __resize_image(self, img, height):
        orig_h, orig_w = img.shape[:2]
        width = int((height / orig_h) * orig_w)      # get width with the same aspect ratio
        return cv2.resize(img, (width, height))

    def __get_temp_img_name(self):
        temp_img_name = os.path.join(TEMP_IMAGE_DIR, f'tmp_img{self.temp_img_cnt}.jpg')
        self.temp_img_cnt += 1
        return temp_img_name


if __name__ == '__main__':

    adr_table_detector = AdrTableDetector(args['threshold'])
    number_detector = NumberDetector(args['threshold'])
    converter = NumberToSubstanceConv('substances_files/Materija.json', 'substances_files/IdentifikacijaOpasnosti.json')

    if args['validation_test']:
        report_generator = ReportGenerator()
        correct_cnt = 0
        total_cnt = 0

        for image_path in os.listdir(args['image_dir']):

            if image_path.endswith('.jpg'):
                report_generator.add_heading(image_path)
                print(image_path)

                base_name = os.path.splitext(image_path)[0]
                img = cv2.imread(os.path.join(args['image_dir'], image_path))

                adr_table = adr_table_detector.detect_adr_table(img)
                report_generator.add_image(adr_table)
                adr_table = cv2.medianBlur(adr_table, 1)    # todo delete or move

                first_row_images, second_row_images = number_detector.detect_numbers(
                    adr_table,
                    ImageUtil.no_extension_file_name(image_path))
                report_generator.add_number_images(first_row_images + second_row_images)

                first_row, second_row = number_detector.get_rows(first_row_images, second_row_images)
                report_generator.add_text(f'First row: {first_row}')
                report_generator.add_text(f'Second row: {second_row}')

                report_generator.add_text(f'Danger: {converter.get_danger_name(first_row)}')
                report_generator.add_text(f'Substance: {converter.get_substance_name(second_row)}')

                with open(f'{args["ground_truth_dir"]}/{base_name}.json') as f:
                    ground_truth = json.load(f)
                if first_row == ground_truth['first_row'] and second_row == ground_truth['second_row']:
                    correct_cnt += 1
                total_cnt += 1

        accuracy = correct_cnt / total_cnt
        report_generator.add_heading('Test results: ')
        report_generator.add_text(f'Correct: {correct_cnt}')
        report_generator.add_text(f'Total: {total_cnt}')
        report_generator.add_text(f'Accuracy: {format(round(accuracy, 3) * 100, ".1f")}%')
        report_generator.generate_report(os.path.join(OUT_DIR, 'TestReport'))

    else:
        logging.info("entered else")
        results_dict = {'results': []}

        for image_path in os.listdir(args['image_dir']):

            if image_path.endswith('.jpg'):
                print(image_path)

                base_name = os.path.splitext(image_path)[0]
                img = cv2.imread(os.path.join(args['image_dir'], image_path))

                adr_table = adr_table_detector.detect_adr_table(img)
                adr_table = cv2.medianBlur(adr_table, 1)  # todo delete or move

                first_row_images, second_row_images = number_detector.detect_numbers(
                    adr_table,
                    ImageUtil.no_extension_file_name(image_path))

                first_row, second_row = number_detector.get_rows(first_row_images, second_row_images)

                results_dict['results'].append({
                    'image_name': image_path,
                    'danger_name': converter.get_danger_name(first_row),
                    'substance_name': converter.get_substance_name(second_row)
                })

        with open(os.path.join(OUT_DIR, INFERENCE_RESULTS_FILE_NAME), "w") as f:
            json.dump(results_dict, f)
