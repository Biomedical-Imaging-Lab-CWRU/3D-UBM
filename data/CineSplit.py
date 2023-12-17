from __future__ import print_function
import numpy as np
# from pathlib import Path
import os
import argparse
import time
from scipy.ndimage.interpolation import zoom
import configparser
import nrrd
import glob
import nibabel as nib
from tqdm import tqdm
import skvideo.io
import re

import sys
sys.tracebacklimit = 0


class DataStruct:
    def __init__(self):
        self.image = None
        self.spacing_x = None
        self.spacing_y = None
        self.spacing_z = None
        self.temporal = None
        self.nasal = None
        self.orientation = None
        self.slices = None
        self.eye = None
        self.date = None
        self.comment = None
        self.reverse_frames = False
        self.old_cine = False


class ArgStruct:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Convert Aviso Cineloop files to standard image/video formats.")
        parser.add_argument('--nocrop', action='store_true', help="Suppress loose cropping of images.")
        rescale_args = parser.add_mutually_exclusive_group()
        rescale_args.add_argument('--stretch-rows', action='store_true', help="Upsample rows to produce square pixels.")
        rescale_args.add_argument('--shrink-cols', action='store_true', help="Downsample columns to produce square pixels.")
        parser.add_argument('--resample-xy', metavar="k", type=float, default=1, help="Resample xy frames by factor k after stretching/shrinking.")
        parser.add_argument('--old-cine', action='store_true', help="Change row size to 256 for old Cineloop files.")
        parser.add_argument('--allow-no-log', action='store_true', help="Allow program to proceed with default values if no log file is found.")
        parser.add_argument('cinefile', nargs='?', help="Input filename. If not specified, program will look for a single .cin file in the current directory.")
        parser.add_argument('logfile', nargs='?', help="Log filename. If not specified, program will look for a single .log file in the current directory. If there is no file, default parameters will be used.")
        parser.add_argument('--keep-slices', metavar="n", type=int, default=9999, help="Keep n xy frames out of stack (default = all")
        parser.add_argument('-o', '--outfile', action='append', dest='outfile', help='Output file.', required=True)

        args = vars(parser.parse_args())

        if args["cinefile"] is None:
            print('Cineloop file not specified. Looking in current directory...')
            files = glob.glob("*.cin")
            if len(files) == 1:
                self.cinefile = files[0]
                # print('Using ', self.cinefile)
            else:
                raise RuntimeError('Could not determine Cineloop file. Make sure there is exactly one .cin file in this'
                                   ' directory or specify it on the command line.')
        else:
            self.cinefile = args["cinefile"]
        print('Using Cineloop:', self.cinefile)

        if args["logfile"] is None:
            cine_directory = os.path.split(os.path.abspath(self.cinefile))
            files = glob.glob(cine_directory[0] + "/" + "*.log")
            if len(files) == 1:
                self.logfile = files[0]
                print('Using log:', self.logfile)
            else:
                print('Could not determine log file.')
                if args["allow_no_log"]:
                    print('DEFAULT VALUES WILL BE USED.')
                    self.logfile = None
                else:
                    print('Either specify log file location or use --allow-no-log.')
                    quit()
        else:
            self.logfile = args["logfile"]

        self.crop_frames = not args["nocrop"]
        self.resample = args["resample_xy"]

        if args["stretch_rows"]:
            self.scale = "rows"
        elif args["shrink_cols"]:
            self.scale = "cols"
        else:
            self.scale = None

        self.keep_slices = args["keep_slices"]
        self.old_cine = args["old_cine"]
        self.outfile=args["outfile"]


def cin_to_array(input_file, data, old_cine):
    """
    Convert data in input_file to a numpy array.
    :param input_file: name of file to read
    :param data: structure to populate with data
    :param old_cine: flag to tell whether old cineloop dimensions should be used
    :return:
    """
    print(my_header("Cineloop parser"))
    print('%30s %s' % ('Parsing file:', input_file))
    dt = np.uint8
    raw_data = np.fromfile(input_file, dtype=dt)
    cin_size_bytes = len(raw_data)
    print('%30s %d' % ('File size:', cin_size_bytes))
    print(np.amax(raw_data))
    
    
    # Header size seems fixed
    cin_header_bytes = 8
    print('%30s %d' % ('File header size:', cin_header_bytes))

    # Actual image size seems fixed
    row_bytes = 2048
    if old_cine:
        num_rows = 256
    else:
        num_rows = 384

    print('%30s %dx%d' % ('Frame dimensions:', num_rows, row_bytes))

    # Determine number of frames and frame header size
    # Loop over possible combinations until we hit the right file size
    # TODO allow multiple hits rather than stopping after the first one
    num_frames = 0
    frame_header_bytes = 0
    for i_frame in range(1, 1505):
        for i_header in range(500, 900):
            if i_frame * (row_bytes*num_rows + i_header) + cin_header_bytes == cin_size_bytes:
                num_frames = i_frame
                frame_header_bytes = i_header
                break
        if num_frames > 0:
            break

    if num_frames == 0:
        raise RuntimeError("Cineloop file size is incompatible with search parameters.")

    segment_bytes = row_bytes * num_rows + frame_header_bytes
    print('%30s %d' % ('Segment size:', segment_bytes))
    print('%30s %d' % ('Segment header size:', frame_header_bytes))
    print('%30s %d' % ('Number of frames:', num_frames))

    # this_frame = np.zeros(shape=(num_rows, row_bytes))
    # Note that frame_array holds the rotated frames
    frame_array = np.zeros(shape=(num_frames, row_bytes, num_rows), dtype=dt)
    
    for i_frame in range(num_frames):
        start_byte = cin_header_bytes + (i_frame * segment_bytes) + frame_header_bytes
        stop_byte = start_byte + row_bytes * num_rows
        this_frame = np.reshape(raw_data[start_byte:stop_byte], (num_rows, row_bytes))
        frame_array[i_frame] = np.rot90(this_frame, -1)

    # Important!
    # The data array stores data in the following orientation: (R-L, A-P, I-S)
    # This needs to be modified for NRRD output
    # TODO allow data to be recorded R-L or L-R
    if arg.crop_frames:
        # These are array indices, not pixels
        data.image = frame_array[::-1, 400:1500, :]
    else:
        data.image = frame_array[::-1,:,:]


# def nrrd_export(outfile, data):
#     # Fix axis order
#     fix_data = np.swapaxes(data.image, 2, 0)
#     nrrd_dict = {
#         'encoding': 'raw',
#         'space': 'left-posterior-superior',
#         'space directions': [[data.spacing_x, 0, 0], [0, data.spacing_y, 0], [0, 0, data.spacing_z]],
#         'space origin': (0, 0, 0),
#         'kinds': ('domain',) * 3,
#         'space units': ('"mm"',) * 3,
#         # 'endian': 'little',
#         'keyvaluepairs': {
#             'eye': data.eye,
#             'date': data.date,
#             'comment': data.comment
#         }
#     }
#     nrrd.write(outfile, fix_data, options=nrrd_dict)


def movie_export(outfile, data):
    skvideo.io.vwrite(outfile, data.image, inputdict={'-r': '10'}, outputdict={'-r': '10', '-c:v':'libx264', '-profile:v':'baseline', '-level':'3.0', '-pix_fmt':'yuv420p', '-g':'8'}, verbosity=1)


def nifti_export(outfile, data):
    fix_data = np.swapaxes(data.image, 2, 0)
    mat_scale = np.diag([data.spacing_x, data.spacing_y, data.spacing_z, 1])
    mat_rot = [[0,0,1,1],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]]
    affine = np.matmul(mat_rot, mat_scale)
    # affine = mat_scale
    array_img = nib.Nifti1Image(fix_data, affine)
    nib.save(array_img, outfile)


def parse_log(logfile, data):
    print(my_header("Log file"))
    print('%30s %s' % ('Reading log file:', logfile))
    config = configparser.ConfigParser()
    config.read(logfile)
    data.slices = config.getint('EyeScan', 'slices')
    data.spacing_x = config.getfloat('EyeScan', 'x_spacing')
    data.spacing_y = config.getfloat('EyeScan', 'y_spacing')
    data.spacing_z = config.getfloat('EyeScan', 'z_spacing')
    data.comment = config.get('EyeScan', 'comment')
    data.eye = config.get('EyeScan', 'eye')
    data.temporal = config.getfloat('EyeScan', 'temporal_position')
    data.nasal = config.getfloat('EyeScan', 'nasal_position')
    data.date = config.get('EyeScan', 'date')
    try:
        data.reverse_frames = config.getboolean('EyeScan', 'reverse_frames')
    except:
        data.reverse_frames = False


    print('%30s %s' % ('Date:', data.date))
    print('%30s %s' % ('Comment:', data.comment))
    print('%30s %s' % ('Eye:', data.eye))
    print('%30s %s' % ('Frame reverse:', data.reverse_frames))
    print('%30s %s' % ('Slices:', data.slices))
    print('%30s %5.3f, %5.3f, %5.3f' % ('Original slice spacing (mm):', data.spacing_x, data.spacing_y, data.spacing_z))


def my_header(s):
    return "\n" + (" "+s+" ").center(60, '=')


if __name__ == '__main__':
    program_start = time.time()

    # Structure for data and metadata
    data = DataStruct()

    # Structure for arguments
    arg = ArgStruct()

    print(my_header("Config"))
    print('%30s %s' % ('Cineloop file:', arg.cinefile))
    print('%30s %s' % ('Log file:', arg.logfile))
    print('%30s %s' % ('Row upsampling:', "enabled" if arg.scale == "rows" else "disabled"))
    print('%30s %s' % ('Column downsampling:', "enabled" if arg.scale == "cols" else "disabled"))
    print('%30s %s' % ('Cropping:', "enabled" if arg.crop_frames else "disabled"))
    print('%30s %4.2f' % ('Resampling:', arg.resample))
    print('%30s %d' % ('Keep slices:', arg.keep_slices))


    # Check if log file exists and, if not, store default values
    if arg.logfile is None:
        print("Log file not specified. Using default parameters.")
        data.spacing_x = 0.041825095
        data.spacing_y = 0.009582712
        data.spacing_z = 0.018
        data.comment = "Processed without log file. Default spacing used."
    else:
        if not os.path.isfile(arg.logfile):
            raise RuntimeError('Specified log file (' + arg.logfile + ') does not exit.')
        else:
            parse_log(arg.logfile, data)

    # Check if Cine file exists
    if not os.path.isfile(arg.cinefile):
        raise RuntimeError('Specified cine file (' + arg.cinefile + ') does not exit.')

    # Extract pixel data from Cineloop file
    cin_to_array(arg.cinefile, data, arg.old_cine)

    print(my_header("Resampling"))

    zoom_x = zoom_y = zoom_z = 1

    # Keep specified slices
    orig_slices = len(data.image)
    if arg.keep_slices < orig_slices:
        slices = [int(round(x)) for x in np.linspace(0, orig_slices - 1, arg.keep_slices)]
        # print(slices)
        data.image = data.image[slices]
        data.spacing_z *= orig_slices / arg.keep_slices
        print("Keeping %d out of %d slices." % (arg.keep_slices, orig_slices))
    else:
        print("Keeping all %d slices." % orig_slices)

    # Compute x and y scale factors for row expansion and downsampling
    if arg.scale or (arg.resample != 1):
        start_time = time.time()

        # Square up pixels
        box_ratio = (11. / 263.) / (6.43 / 671)
        if arg.scale == "rows":
            zoom_x *= box_ratio
        elif arg.scale == "cols":
            zoom_y /= box_ratio

        # Keep these in (z, y, x) order
        zoom_x *= arg.resample
        zoom_y *= arg.resample
        # For now don't zoom z

        # Scale data
        # Remember that the z-axis is first in the data array
        print('\nResampling with factors %.3f, %.3f, %.3f' % (zoom_x, zoom_y, zoom_z))

        # Resample one frame to determine size for pre-allocated output array
        dummy_image = zoom(data.image[0], (zoom_y, zoom_x), order=3)
        zoom_data_shape = np.append([data.image.shape[0]], dummy_image.shape)
        zoom_data = np.empty(zoom_data_shape, dtype=np.uint8)

        for i in tqdm(range(data.image.shape[0])):
            zoom_data[i] = zoom(data.image[i], (zoom_y, zoom_x), order=3)

        data.image = zoom_data

        # Old way
        # data.image = zoom(data.image, (zoom_z, zoom_y, zoom_x), order=3)

        # Adjust voxel spacing
        data.spacing_x /= zoom_x
        data.spacing_y /= zoom_y
        data.spacing_z /= zoom_z

        end_time = time.time()
        print('...done (%.2f seconds)' % (end_time-start_time))
    else:
        print('No resampling of frames.')

    print('%30s %5.3f, %5.3f, %5.3f' % ('New slice spacing (mm):', data.spacing_x, data.spacing_y, data.spacing_z))

    # Base for output file(s). Add suffix later.
    # outfile_base = Path(arg.cinefile).stem

    # outfile = outfile_base + ".nrrd"
    # print('Output file: %s' % outfile)
    # nrrd_export(outfile, data)

    # outfile = outfile_base + ".dicom"
    # print('Output file: %s' % outfile)
    # dicom_export(outfile, data)

    # outfile = outfile_base + ".nii"
    for this_output in arg.outfile:
        print('Output file: %s' % this_output)
        if re.match('.*\.(avi|mp4)', this_output):
            movie_export(this_output, data)
        elif re.match('.*\.nii', this_output):
            print('Writing NIFTI format')
            nifti_export(this_output, data)
        # elif re.match('.*\.nrrd', this_output):
        #     print('Writing NRRD format')
        #     nrrd_export(this_output, data)
        else:
            print('Could not guess output format.')

    end_time = time.time()
    print('\nCineSplit done. (%.2f sec)' % (end_time - program_start))
