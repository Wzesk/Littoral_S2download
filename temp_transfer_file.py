#import tario

# import io
# from io import BytesIO
# import tarfile
# import datetime
# import os
# import json
# import pandas as pd
# from PIL import Image
# from skimage.morphology import binary_dilation, disk, remove_small_objects, binary_erosion
# import tarfile
# import requests
# from pathlib import Path
# from functools import partial
# import rasterio as rio
# import numpy as np
# from matplotlib import pyplot as plt

# import omnicloudmask
# from omnicloudmask import (
#     predict_from_load_func,
#     predict_from_array,
#     load_s2,
# )


# try:
#     import geemap, ee
# except ModuleNotFoundError:
#     if 'google.colab' in str(get_ipython()):
#         print("package not found, installing w/ pip in Google Colab...")
#         !pip install geemap
#     else:
#         print("package not found, installing w/ conda...")
#         !conda install mamba -c conda-forge -y
#         !mamba install geemap -c conda-forge -y
#     import geemap, ee

# try:
#         ee.Initialize(project='ee-shorelinetracker')
# except Exception as e:
#         ee.Authenticate()
#         ee.Initialize(project='ee-shorelinetracker')


##################################################
# #############  ee functions ####################
##################################################


# def get_image_collection(data,cloudy_pixel_percentage=10):
#   aoi_rec = ee.Geometry.Rectangle(data["aoi"])
#   se2_col = ee.ImageCollection('COPERNICUS/S2').filterDate(data["start_date"],data["end_date"]).filterBounds(aoi_rec).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage))
#   return se2_col

# def visualize_nir_from_collection(data, se2_col,index):
#   aoi = ee.Geometry.Rectangle(data["aoi"])

#   rg_nir = ['B4','B3','B8'] # blue is 'B2'
#   rg_nir_img = ee.Image(se2_col.toList(se2_col.size()).get(index)).select(rg_nir).clip(aoi)
#   rg_nir_img_disp = rg_nir_img.divide(10000)
#   return rg_nir_img_disp

# def retrieve_rgb_nir_from_collection(data, se2_col,index):
#   aoi = ee.Geometry.Rectangle(data["aoi"])

#   rg_nir = ['B4','B3','B8'] # blue is 'B2'
#   rg_nir_img = ee.Image(se2_col.toList(se2_col.size()).get(index))#.select(rg_nir)
#   url = rg_nir_img.getDownloadUrl({
#     'bands': rg_nir,
#     'region': aoi,
#     'scale': 10,
#     'format': 'NPY'
#   })
#   projection = rg_nir_img.select('B8').projection().getInfo()
#   response = requests.get(url)
#   data = np.load(io.BytesIO(response.content))

#   # stack 3 2d arrays into 3d array
#   img_arr = np.dstack([data['B4'], data['B3'], data['B8']])
#   #move 3 dimension to first position
#   img_arr = np.moveaxis(img_arr, -1, 0)
#   # Predict cloud and cloud shadow masks
#   pred_mask = predict_from_array(img_arr)

#   #merge cloud mask with cloud shadow mask
#   pred_array = np.where(pred_mask == 0, 1, 0) + np.where(pred_mask == 2, 1, 0)
#   pred_array = np.where(pred_array > 0, 1, 0)
#   cld_pred = np.squeeze(pred_array)
#   # cld_pred = np.ones_like(img_arr[0,:,:]) # to turn off cloud mask

#   usable_pixels = np.sum(cld_pred) / cld_pred.size



#   return data['B8'], cld_pred,usable_pixels,projection


# def dilate_mask(mask, size_disk=9):
#     mask = binary_dilation(mask, disk(size_disk))
#     ## convert back to PIL format
#     mask_image = Image.fromarray(mask)
#     return mask_image

# def process_collection_images(data, se2_col):
#   aoi = ee.Geometry.Rectangle(data["aoi"])
#   name = data["project_name"]
#   path = data["path"]
#   threshold = data["usable_pixel_percentage"]
#   length = se2_col.size().getInfo()

#   #if path does not exist, creat it
#   if not os.path.exists(path):
#     os.makedirs(path)
#   rgnir_path = path + "/rgnir.tar"
#   cldmsk_path = path + "/cldmsk.tar"

#   #create a new pandas dataframe with the columns: Index, name, status,usable_percentage
#   proj_track = pd.DataFrame(columns=['Index', 'name', 'rgnir_download','cld_calculation','percentage_usable'])


#   for i in range(length):
#     nir,cld,percentage_usable,projection = retrieve_rgb_nir_from_collection(data, se2_col, i)
#     print("percentage usable pixels = " + str(percentage_usable))

#     #get the name of the i image file from the collection
#     img_name = ee.Image(se2_col.toList(se2_col.size()).get(i)).get('system:index').getInfo()
#     new_row = pd.DataFrame({'Index': [i], 'name': [img_name], 'rgnir_download': [False], 'cld_calculation': [False], 'percentage_usable': [percentage_usable]})

#     if(percentage_usable > threshold):
#       # # saving projection info if needed
#       # json_path = os.path.join(path, img_name + ".json")
#       # with open(json_path, 'w') as f:
#       #   json.dump(projection, f)

#       #normalize
#       n_nir = (nir - np.min(nir))/(np.max(nir) - np.min(nir))
#       #stack
#       nir_3d = np.dstack((n_nir,n_nir,n_nir))
#       nir_img = Image.fromarray((nir_3d * 255).astype(np.uint8))
#       #normalize
#       cld = cld/np.max(cld)
#       cld_3d = np.dstack((cld,cld,cld))
#       cld_img = Image.fromarray((cld_3d * 255).astype(np.uint8))

#       save_to_tar(rgnir_path,nir_img,img_name + "_rgnir.png",overwrite=True)
#       new_row['rgnir_download'] = True

#       save_to_tar(cldmsk_path,cld_img,img_name + "_cld.png",overwrite=True)
#       new_row['cld_calculation'] = True

#       print(img_name + " saved")
#     else:
#       print(str(i) + " has too many clouds")

#     # add row to table

#     proj_track = pd.concat([proj_track, new_row], ignore_index=True)

#   #save proj_track to csv
#   proj_track.to_csv(path + "/" + "proj_track.csv", index=False)

#   name = data["project_name"]
#   date = datetime.datetime.now()
#   date_str = date.strftime("%Y-%m-%d")
#   set_last_run(name, date_str)
#   print("finished run:",name,date_str)

#   return proj_track


# # def removeLandsatClouds(image):
# #   cloudShadowBitMask = ee.Number(2).pow(3).int()
# #   cloudsBitMask = ee.Number(2).pow(5).int()
# #   qa = image.select('pixel_qa')
# #   mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
# #   return image.updateMask(mask)



# def save_parameters_to_json(aoi, start, end, usable_pixel_percentage, project_name, path):
#   data = {
#       "aoi": aoi,
#       "start_date": start,
#       "end_date": end,
#       "usable_pixel_percentage": usable_pixel_percentage,
#       "project_name": project_name,
#       "path":path
#   }
#   save_path = path + "/" + project_name + ".json"
#   with open(save_path, 'w') as json_file:
#       json.dump(data, json_file, indent=4)

#   return data


# def image_to_tar_format(img, image_name):
#     buff = BytesIO()
#     if '.png' in image_name.lower():
#         img = img.convert('RGBA')
#         img.save(buff, format='PNG')
#     else:
#         img.save(buff, format='JPEG')
#     buff.seek(0)
#     fp = io.BufferedReader(buff)
#     img_tar_info = tarfile.TarInfo(name=image_name)
#     img_tar_info.size = len(buff.getvalue())
#     return img_tar_info, fp

# def get_tar_filenames(tar_path):
#   tar = tarfile.open(tar_path, 'r')
#   names = []
#   members = tar.getmembers()
#   for member in members:
#     names.append(member.name)
#   tar.close()
#   return names

# def get_from_tar(tar_path,name):
#   tar = tarfile.open(tar_path, 'r')
#   members = tar.getmembers()
#   for member in members:
#     if member.name == name:
#       img_bytes = BytesIO(tar.extractfile(member.name).read())
#       img = Image.open(img_bytes, mode='r')
#       tar.close()
#       return img
#   # no image was found with that name
#   print(name + " was not in tar.")
#   tar.close()
#   return None

# def save_to_tar(tar_path,img,img_name,overwrite=False):
#   if os.path.exists(tar_path):
#     tar = tarfile.open(tar_path, 'r')
#     members = tar.getmembers()
#     if len(members) > 0 :
#       for member in members:
#         if member.name == img_name:
#           print(overwrite)
#           if not overwrite:
#             print(img_name+" exists, skipping")
#             tar.close()
#             return tar_path
#           else:
#             print(img_name+" exists, overwriting")
#     else:
#       print("empty tar archive, adding: "+ img_name)
#     tar.close()
#     save_tar = tarfile.open(tar_path, 'a')
#     img_tar_info, fp = image_to_tar_format(img, img_name)
#     save_tar.addfile(img_tar_info, fp)
#     save_tar.close()
#     save_tar.close()
#   else:
#     print("starting new tar archive with: "+ img_name)
#     # save file to tar
#     save_tar = tarfile.open(tar_path, 'w')
#     img_tar_info, fp = image_to_tar_format(img, img_name)
#     save_tar.addfile(img_tar_info, fp)
#     save_tar.close()

#   return tar_path


##################################################
# #############  ee functions ####################
##################################################


# def load_sites():
#   # Replace 'Your spreadsheet name' with the actual name of your Google Sheet
#   spreadsheet = gc.open('littoral_analysis_sites')
#   # Replace 'Sheet1' with the name of the sheet you want to read
#   worksheet = spreadsheet.worksheet('sites')
#   # Get all values from the worksheet
#   rows = worksheet.get_all_values()
#   # Convert the data to a Pandas DataFrame
#   import pandas as pd
#   df = pd.DataFrame.from_records(rows[1:], columns=rows[0]) # Assuming the first row is the header
#   # Print the DataFrame
#   return df

# def get_site_by_name(name):
#   sites = load_sites()
#   site = sites[sites['site_name'] == name]
#   return site

# def list_site_names():
#   sites = load_sites()
#   return sites['site_name'].tolist()

# def set_last_run(site_name, date):
#   sites = load_sites()
#   sites.loc[sites['site_name'] == site_name, 'last_run'] = date
#   spreadsheet = gc.open('littoral_analysis_sites')
#   worksheet = spreadsheet.worksheet('sites')
#   worksheet.update([sites.columns.values.tolist()] + sites.values.tolist())
#   new_sites = load_sites()
#   return new_sites


# def load_site_parameters(name,path):
#   site_row = get_site_by_name(name)
#   aoi_str = site_row['aoi'].values[0]
#   aoi_str = "".join(aoi_str.split())
#   aoi = json.loads(aoi_str)
#   aoi_rec = ee.Geometry.Rectangle(aoi)

#   start = site_row['start'].values[0]
#   end = site_row['end'].values[0]
#   usable_percentage = float(site_row['usable_percentage'].values[0])
#   proj_name = name
#   path = path

#   save_path = path + "/" + proj_name
#   if not os.path.exists(save_path):
#     os.makedirs(save_path)

#   proj_data = save_parameters_to_json(aoi, start, end, usable_percentage, proj_name, save_path)

#   return proj_data