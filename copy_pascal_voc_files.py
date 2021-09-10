# Separate test_images to test_yolo and test_pascal_voc
import os
import shutil
import sys
directory = sys.argv[1]
destination_folder = sys.argv[2]
file_paths = []
for filename in os.listdir(directory):
    f = os.path.join(directory,filename)
    if os.path.isfile(f):
        if f.endswith(".xml"):
            # PASCAL VOC Format
            file_without_extension = f[:-4]
            print(file_without_extension)
            # Copy the xml and the corresponding .jpg in train_images_pascal_voc
            xml_file = file_without_extension+".xml"
            jpg_file = file_without_extension+".jpg"
            shutil.copy(xml_file,destination_folder)
            shutil.copy(jpg_file,destination_folder)

            
