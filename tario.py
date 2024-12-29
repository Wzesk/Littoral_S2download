import io
from io import BytesIO
import tarfile
from PIL import Image
import os

class tar_io:
    def __init__(self,path):
        self.path = path

    def image_to_tar_format(self, img, image_name):
        buff = BytesIO()
        if '.png' in image_name.lower():
            img = img.convert('RGBA')
            img.save(buff, format='PNG')
        else:
            img.save(buff, format='JPEG')
        buff.seek(0)
        fp = io.BufferedReader(buff)
        img_tar_info = tarfile.TarInfo(name=image_name)
        img_tar_info.size = len(buff.getvalue())
        return img_tar_info, fp

    def get_tar_filenames(self):
        tar = tarfile.open(self.path, 'r')
        names = []
        members = tar.getmembers()
        for member in members:
            names.append(member.name)
        tar.close()
        return names

    def get_from_tar(self,name):
        tar = tarfile.open(self.path, 'r')
        members = tar.getmembers()
        for member in members:
            if member.name == name:
                img_bytes = BytesIO(tar.extractfile(member.name).read())
                img = Image.open(img_bytes, mode='r')
                tar.close()
                return img
        # no image was found with that name
        print(name + " was not in tar.")
        tar.close()
        return None

    def save_to_tar(self, img, img_name,overwrite=False):

        if os.path.exists(self.path):
            tar = tarfile.open(self.path, 'r')
            members = tar.getmembers()
            if len(members) > 0 :
                for member in members:
                    if member.name == img_name:
                        if not overwrite:
                            print(img_name+" exists, skipping")
                            tar.close()
                            return self.path
                        else:
                            print(img_name+" exists, overwriting")
            else:
                print("empty tar archive, adding: "+ img_name)

            tar.close()
            save_tar = tarfile.open(self.path, 'a')
            img_tar_info, fp = self.image_to_tar_format(img, img_name)
            save_tar.addfile(img_tar_info, fp)
            save_tar.close()
        else:
            print("starting new tar archive with: "+ img_name)
            # save file to tar
            save_tar = tarfile.open(self.path, 'w')
            img_tar_info, fp = self.image_to_tar_format(img, img_name)
            save_tar.addfile(img_tar_info, fp)
            save_tar.close()

        return self.path