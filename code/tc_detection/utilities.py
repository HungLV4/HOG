import cv2
import csv
import os

def getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, mn):
	return "{:0>4d}_{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}.tir.01.fld".format(bt_ID, yyyy, mm, dd, hh, mn)

def getFilePathFromTime(prefix, tc_type, bt_ID, yyyy, mm, dd, hh, mn):
	# return prefix + "%d/" % tc_type + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, mn) + ".png"
	return prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, mn) + ".png"

def latlon2xy(lat, lon):
	if lat > 60 or lat < -60 or lon > 205 or lon < 85:
		return -1, -1

	return int((60 - lat) / 0.067682), int((lon - 85) / 0.067682)

""" Prepare the training images by cropping around best track point
"""
def cropImagesByBestTrack(im_full_prefix, im_area_prefix, full_bt_filepath, area_bt_filepath):
	with open(full_bt_filepath, 'rb') as full_file, open(area_bt_filepath, 'wb') as area_file:
		reader = csv.reader(full_file, delimiter=',')
		writer = csv.writer(area_file, delimiter=',')
		
		index = 0
		for row in reader:
			print "Processing TC:", row[7]
			bt_ID = int(row[1])
			numOfDataLines = int(row[2])
			for i in range(numOfDataLines):
				line = reader.next()
				
				tc_type = int(line[2])

				datetime = line[0]

				yyyy = 2000 + int(datetime[0:2])
				mm = (int)(datetime[2:4])
				dd = (int)(datetime[4:6])
				hh = (int)(datetime[6:8])

				impath = getFilePathFromTime(im_full_prefix, bt_ID, yyyy, mm, dd, hh, 00)
				ref_impath = getFilePathFromTime(im_full_prefix, bt_ID, yyyy, mm, dd, hh, 10)
				
				if not (os.path.isfile(impath) and os.path.isfile(ref_impath)):
					continue

				im = cv2.imread(impath, 0)
				ref_im = cv2.imread(ref_impath, 0)

				height, width = im.shape

				bt_lat = int(line[3]) * 0.1
				bt_lon = int(line[4]) * 0.1

				row, col = latlon2xy(bt_lat, bt_lon)

				w = 256
				if row - w > 0 and row + w + 1 < height and col - 1 > 0 and col + w + 1 < width:
					crop_im = im[row - w : row + w , col - w : col + w]
					crop_ref = ref_im[row - w : row + w, col - w : col + w]
					
					cv2.imwrite(getFilePathFromTime(im_area_prefix, tc_type, bt_ID, yyyy, mm, dd, hh, 00), crop_im)
					cv2.imwrite(getFilePathFromTime(im_area_prefix, tc_type, bt_ID, yyyy, mm, dd, hh, 10), crop_ref)

					writer.writerow([bt_ID, datetime, tc_type])