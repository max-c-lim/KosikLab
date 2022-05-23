import json
import h5py
import os
import glob
import copy
import sys
import subprocess

import braingeneers.utils.s3wrangler as wr
import braingeneers.utils.smart_open_braingeneers as smart_open

from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QLineEdit, QFileDialog, QPushButton, QLabel, QTextEdit
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QIcon

# filename = '210626_Stim2.raw.h5'
# descriptor = 'Test'
# notes = 'Test_notes'

# with h5py.File(filename, "r") as f:
#     # List all groups
#     print("Keys: %s" % f.keys())
#     keys = list(f.keys())
#     a_group_key = list(f.keys())[0]
#     # Get the data
#     data = list(f[a_group_key])
#     x = f.get('sig')[()]
#     settings = f.get('time')[()]
#     mapping = f.get('mapping')[()]


s3 = 'aws --endpoint https://s3.nautilus.optiputer.net s3'
s3_path = 'braingeneers/ephys/'


class App(QWidget):

    def __init__(self,main_app = None):
        super().__init__()
        self.main_app = main_app
        self.title = 'MaxWell S3 Experiment Upload'
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 480

        self.button1 = None
        self.button2 = None
        self.button3 = None

        self.label1 = None
        self.label2 = None
        self.label3 = None
        self.label4 = None
        self.label5 = None

        self.textbox1 = None
        self.textboxlabel1 = None
        self.textbox2 = None
        self.textboxlabel2 = None

        self.infobutton = None

        self.dir_name = 'No Directory Selected'
        self.exp_list = ['No experiments chosen yet']
        self.uuid_prefix = ''

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.button1 = QPushButton(self)
        self.button1.move(0, 5)
        self.button1.setText("Open Directory")
        self.button1.clicked.connect(self.button1_clicked)

        self.button2 = QPushButton(self)
        self.button2.move(0, 60)
        self.button2.setText("Generate Metadata")
        self.button2.clicked.connect(self.button2_clicked)
        self.button2.setEnabled(False)

        self.button3 = QPushButton(self)
        self.button3.move(0, 360)
        self.button3.setText("Upload")
        self.button3.clicked.connect(self.button3_clicked)
        self.button3.setEnabled(False)

        self.infobutton = QPushButton(self)
        self.infobutton.move(140, 360)
        self.infobutton.setText("Help")
        self.infobutton.clicked.connect(self.infobutton_clicked)

        self.label1 = QLabel(self)
        self.label1.setFixedWidth(640)
        self.label1.move(20, 40)
        self.label1.setText(self.dir_name)

        self.label2 = QLabel(self)
        # self.label2.setFixedWidth(200)
        # self.label2.setFixedHeight(20)

        self.label2.move(220, 20)
        self.label2.resize(600, 400)
        self.label2.setText('\n'.join(self.exp_list))

        self.label3 = QLabel(self)
        self.label3.setFixedWidth(400)
        self.label3.move(220, 0)
        font = self.label3.font()
        font.setBold(True)
        self.label3.setFont(font)
        self.label3.setText('Experiments:')

        self.label4 = QLabel(self)
        self.label4.setFixedWidth(400)
        self.label4.move(20, 400)
        self.label4.setText('')

        self.label5 = QLabel(self)
        self.label5.setFixedWidth(400)
        self.label5.move(20, 420)
        self.label5.setText('')

        self.textboxlabel1 = QLabel(self)
        self.textboxlabel1.setFixedWidth(400)
        self.textboxlabel1.move(20, 90)
        self.textboxlabel1.setText('UUID:')

        self.textbox1 = QLineEdit(self)
        self.textbox1.move(20, 110)
        self.textbox1.resize(180, 20)
        self.textbox1.setEnabled(False)
        self.textbox1.setText("None")

        self.textboxlabel2 = QLabel(self)
        self.textboxlabel2.setFixedWidth(400)
        self.textboxlabel2.move(20, 130)
        self.textboxlabel2.setText('Notes:')

        self.textbox2 = QTextEdit(self)
        self.textbox2.move(20, 150)
        self.textbox2.resize(180, 200)
        self.textbox2.setEnabled(False)
        self.textbox2.setText("")



        self.show()

    def button1_clicked(self):
        self.open_directory_dialog()
        self.button2.setEnabled(True)
        self.uuid_prefix = get_uuid_prefix(self.dir_name, self.exp_list)
        self.textbox1.setText(self.uuid_prefix)
        self.textbox1.setEnabled(True)
        self.textbox2.setEnabled(True)

    def button2_clicked(self):
        print('Generating metadata...')

        self.uuid = self.textbox1.text()
        self.notes = self.textbox2.toPlainText()

        gen_metadata(self.dir_name, self.exp_list, uuid=self.uuid, notes=self.notes)
        gen_metadata_exp(self.dir_name, self.exp_list)

        self.button3.setEnabled(True)

    def button3_clicked(self):
        print('Uploading to S3...')
        self.button3.setText("Uploading...")
        upload_dir_s3(self.uuid, self.dir_name, self.exp_list,app=self)

        # self.label4 = QLabel(self)
        # self.label4.setFixedWidth(400)
        # self.label4.move(0, 380)
        # self.label4.setText('COMPLETE')



    def infobutton_clicked(self):
        '''Give information on how to use this interface'''
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("Welcome to the MaxWell S3 Upload Process!")
        msg.setInformativeText('Please begin by opening a folder locally using the\
                               "Open Directory" button. This folder should include \
                               the desired experiments (data1.raw.h5, data2... etc). \
                               \n\nYou should then see the files populate on the right \
                               side of the window. Please use the prompts to populate the uuid \
                               and the notes, then click "Generate Metadata".\
                               \n\nFinally Click "Upload" and wait until you see "Upload complete".\
                               \n\nHappy uploading!')
        msg.setWindowTitle("S3 Upload information")
        msg.setDetailedText("Boo!")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        retval = msg.exec_()
        # print("value of pressed message box button:", retval)

    def open_directory_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.dir_name = str(
            QFileDialog.getExistingDirectory(self, "Experiment Directory for S3 upload", options=options))

        self.label1.setText(self.dir_name)

        self.exp_list = get_experiments(self.dir_name)
        self.label2.setText('\n'.join(self.exp_list))


# Methods for creating metadata
def get_experiments(exp_path=''):
    print(exp_path)
    exp_list = glob.glob(exp_path + '/' + '*.h5')
    for i, exp in enumerate(exp_list):

        try:
            with h5py.File(exp, "r") as f:
                pass

        except OSError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)

            msg.setText(f'The file {exp} could not be read!')
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            retval = msg.exec_()
            exp_list.remove(exp)

    return exp_list


def get_uuid_prefix(dir_name, exp_list):
    f = h5py.File(exp_list[0], "r")
    keys = list(f.keys())
    time_stamp = str(f.get('time')[()])

    # Correct format
    meta_time_stamp = time_stamp[10:20] + 'T' + time_stamp[21:29]
    f.close()

    uuid = meta_time_stamp[:10] + '-e-'
    return uuid


def gen_metadata(dir_name, exp_list, uuid='', notes=''):
    # Open file
    with h5py.File(exp_list[0], "r") as f:
        keys = list(f.keys())
        time_stamp = str(f.get('time')[()])

        # Correct format
        meta_time_stamp = time_stamp[10:20] + 'T' + time_stamp[21:29]

    metadata = {
        "experiments": [
        ],
        "notes": " ",
        "timestamp": "",
        "uuid": "YYYY-MM-DD-[e]-[descriptor]"
    }

    metadata['timestamp'] = meta_time_stamp
    metadata['uuid'] = uuid
    metadata['notes'] = notes
    metadata['experiments'] = [f"experiment{i + 1}.json" for i in range(len(exp_list))]

    with open(dir_name + '/' + "metadata.json", "w") as outfile:
        json.dump(metadata, outfile, indent=2)


def gen_metadata_exp(dir_name, exp_list):
    '''
    Loop through each experiment file and save corresponding metadata file
    '''
    exp_metadata = get_metadata_exp_template()

    for i, exp in enumerate(exp_list):

        print(i, exp)

        try:
            with h5py.File(exp, "r") as f:
                cur_metadata = copy.copy(exp_metadata)

                # Get keys
                keys = list(f.keys())
                time_stamp = str(f.get('time')[()])
                # Correct time format
                meta_time_stamp = time_stamp[10:20] + 'T' + time_stamp[21:29]

                # Get blocks
                shape = f.get('sig').shape
                num_frames = shape[0] * shape[1]
                print('Total frames * channels', num_frames)

                # Get path
                data_path = exp

                # Set values
                cur_metadata['blocks'][0]['num_frames'] = num_frames
                cur_metadata['blocks'][0]['path'] = exp
                cur_metadata['blocks'][0]['timestamp'] = meta_time_stamp
                cur_metadata['num_channels'] = shape[0]
                cur_metadata['num_voltage_channels'] = shape[0]
                cur_metadata['timestamp'] = meta_time_stamp

                # Write file
                with open(dir_name + '/' + "experiment{}.json".format(i + 1), "w") as outfile:
                    json.dump(cur_metadata, outfile, indent=2)

        except OSError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)

            msg.setText(f'The file {exp} could not be read!')
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            retval = msg.exec_()


def get_metadata_exp_template():
    exp_metadata = {
        "blocks": [
            {
                "num_frames": 0,
                "path": "",
                "source": "",
                "timestamp": ""
            }
        ],
        "channels": [],
        "hardware": "Maxwell",
        "name": "1well-maxwell",
        "notes": "",
        "num_channels": 0,
        "num_current_input_channels": 0,
        "num_voltage_channels": 0,
        "offset": 0,
        "sample_rate": 20000,
        "scaler": 1,
        "timestamp": "",
        "units": "\u00b5V",
        "version": "0.0.1"
    }
    return exp_metadata


def upload_dir_s3(uuid, dir_name, exp_list,app):
    # cmd = f"{s3} sync {dir_name} s3://{s3_path + uuid} "
    # process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()

    # Upload metadata
    wr.upload(local_file=f'{dir_name}/metadata.json', path='s3://braingeneers/ephys/' + uuid + '/metadata.json')

    for i in range(len(exp_list)):
        wr.upload(local_file=dir_name + f'/experiment{i + 1}.json',
                  path=f's3://braingeneers/ephys/{uuid}/original/experiment{i + 1}.json')

    # Upload data files
    for i,exp in enumerate(exp_list):
        app.label4.setText(f'Uploading file {i + 1}/{len(exp_list)}...')
        print(f'Uploading {exp}')
        filename = exp.split('/')[-1]
        print(f's3://braingeneers/ephys/'\
                '{uuid}/original/data/{filename}')

        print(f's3://braingeneers/ephys/{uuid}/original/data/{filename}')

        b = None
        file_size = os.path.getsize(exp)/1000000
        megabyte_count = 0
        with open(exp, 'rb') as local_file, smart_open.open( f's3://braingeneers/ephys/'\
                                                            f'{uuid}/original/data/{filename}', 'wb') as s3file:
            count = 0
            while b != b'':
                #b = local_file.read()
                #app.label5.setText(f'Uploading file of size=\t {file_size:3.1f} MB')
                #app.main_app.processEvents()
                #s3file.write(b)
            
            

                count += 1
                # upload 10MB
                b = local_file.read(10000000)
                # print(f'Wrote line: {count},{len(b)}')
                s3file.write(b)
                megabyte_count += 10
                app.label5.setText(f'{megabyte_count}\t of \t {file_size:3.1f} MB')
                app.main_app.processEvents()

        # wr.upload(local_file=exp, path=f's3://braingeneers/ephys/{uuid}/original/data/{filename}')

    app.label4.setText('Upload Complete!')
    app.button3.setText("Upload")
    print('Uploaded Successfully')


if __name__ == '__main__':
    main_app = QApplication(sys.argv)
    ex = App(main_app)
    sys.exit(main_app.exec_())
