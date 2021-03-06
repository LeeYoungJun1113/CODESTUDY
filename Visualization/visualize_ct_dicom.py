!pip install kaggle --upgrade

# from google.colab import files
# uploaded = files.upload()

# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/ 
# !chmod 600 ~/.kaggle/kaggle.json

# !ls -1ha kaggle.json

from google.colab import drive

ROOT = "/content/drive"
print(ROOT)
drive.mount(ROOT)

# !unzip '*.zip'

!pip install pydicom

import numpy as np
import pandas as pd
from skimage.io import imread
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import os
import pydicom as dicom

PATH_overview = '/content/drive/MyDrive/PRACTICE/input/siim-medical-image'
data_df = pd.read_csv(os.path.join(PATH_overview, "overview.csv"))

print("CT medical iamges - rows:", data_df.shape[0]," columns:", data_df.shape[1])

data_df.head()

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/PRACTICE/input/siim-medical-image/tiff_images/

!unzip -qq '/content/drive/MyDrive/PRACTICE/input/siim-medical-image/tiff_images/*.zip'

PATH_tiff = '/content/drive/MyDrive/PRACTICE/input/siim-medical-image'

print("Number of TIFF images:", len(os.listdir(os.path.join(PATH_tiff, "tiff_images"))))

# tiff_data = pd.DataFrame([{'path': filepath}
#                           for filepath in glob(PATH_tiff+'tiff_images/*.tif')])
tiff_data = pd.DataFrame([{'path': filepath} for filepath in glob(PATH_tiff+'tiff_images/*.tif')])

def process_data(path):
    data = pd.DataFrame([{'path': filepath} for filepath in glob(PATH_tiff+path)])
    data['file'] = data['path'].map(os.path.basename)
    data['ID'] = data['file'].map(lambda x: str(x.split('_')[1]))
    data['Age'] = data['file'].map(lambda x: int(x.split('_')[3]))
    data['Contrast'] = data['file'].map(lambda x: bool(int(x.split('_')[5])))
    data['Modality'] = data['file'].map(lambda x: str(x.split('_')[6].split('.')[-2]))
    return data

tiff_data = process_data('/tiff_images/*.tif')

tiff_data.head(10)

PATH_dicom = '/content/drive/MyDrive/PRACTICE/input/siim-medical-image'
print("Number of DICOM filkes:", len(os.listdir(os.path.join(PATH_dicom, "dicom_dir"))))

dicom_data = process_data('/dicom_dir/*.dcm')

dicom_data.head(10)

def countplot_comparison(feature):
  fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,4))
  s1 = sns.countplot(data_df[feature], ax=ax1)
  s1.set_title("Overview data")
  s2 = sns.countplot(tiff_data[feature], ax=ax2)
  s2.set_title("Tiff files data")
  s3 = sns.countplot(dicom_data[feature], ax=ax3)
  s3.set_title("Dicom files data")
  plt.show()

countplot_comparison('Age')

countplot_comparison('Contrast')

def show_images(data, dim=16, imtype='TIFF'):
    img_data = list(data[:dim].T.to_dict().values())
    f, ax = plt.subplots(4,4, figsize=(16,20))
    for i,data_row in enumerate(img_data):
        if(imtype=='TIFF'): 
            data_row_img = imread(data_row['path'])
        elif(imtype=='DICOM'):
            data_row_img = dicom.read_file(data_row['path'])
        if(imtype=='TIFF'):
            ax[i//4, i%4].matshow(data_row_img,cmap='gray')
        elif(imtype=='DICOM'):
            ax[i//4, i%4].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 
        ax[i//4, i%4].axis('off')
        ax[i//4, i%4].set_title('Modality: {Modality} Age: {Age}\nSlice: {ID} Contrast: {Contrast}'.format(**data_row))
    plt.show()

show_images(tiff_data, 16, 'TIFF')

show_images(dicom_data,16,'DICOM')

dicom_file_path = list(dicom_data[:1].T.to_dict().values())[0]['path']
dicom_file_dataset = dicom.read_file(dicom_file_path)
dicom_file_dataset

print("Modality: {}\nManufacturer: {}\nPatient Age: {}\nPatient Sex:: {}\nPatient Name: {}\nPatient ID: {}".format(
      dicom_file_dataset.Modality,
      dicom_file_dataset.Manufacturer,
      dicom_file_dataset.PatientAge,
      dicom_file_dataset.PatientSex,
      dicom_file_dataset.PatientName,
      dicom_file_dataset.PatientID))



def show_dicom_images(data):
  img_data = list(data[:16].T.to_dict().values())
  f, ax = plt.subplots(4,4, figsize = (16,20))
  for i,data_row in enumerate(img_data):

    data_row_img = dicom.read_file(data_row['path'])
    modality = data_row_img.Modality
    age = data_row_img.PatientAge

    ax[i//4, i%4].imshow(data_row_img.pixel_array, cmap = plt.cm.bone)
    ax[i//4, i%4].axis('off')
    ax[i//4, i%4].set_title('Modality: {} Age: {}\nSlice: {} Contrast: {}'.format(
        modality, age, data_row['ID'], data_row['Contrast']))
  plt.show()

show_dicom_images(dicom_data)

##Plus, The code bellow shows how a Dicom 2D image subset is used to create a 3D scene

# extract voxel data  
def extract_voxel_data(list_of_dicom_files):  
    datasets = [dicom.read_file(f) for f in list_of_dicom_files]  
     try:  
         voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(datasets)  
     except dicom_numpy.DicomImportException as e:  
     # invalid DICOM data  
         raise  
     return voxel_ndarray
