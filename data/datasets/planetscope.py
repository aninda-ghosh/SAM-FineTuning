# encoding: utf-8
"""
@author:  Aninda Ghosh, Manthan Satish
@contact: aghosh57@asu.edu, mcsatish@asu.edu
"""

import numpy as np
import cv2
import geopandas as gpd
import torch
from torch.utils.data import Dataset
from rich.console import Console
from config import cfg

class ParcelDataset(Dataset):
    """Parcel dataset
    --------------------------------
    This class is used to load the dataset.
    The data is stored in a folder with the following structure:
    - data
        - parcels
            - xxxxx.png
            - xxxxx.geojson
            - ...
            - ...
            - parcel_data.geojson

    The labels is loaded using the `geopandas` library.
    The images are loaded using the `matplotlib.pyplot` module.
    The labels are stored in boolean masks. The masks are stored in a list.
    """
    
    def __init__(self, cfg):
        """
        Args:
            path (string): Path to the folder containing the dataset.
            The dataset is stored in a single folder.
        """
        self.cfg = cfg
        self.path = cfg.DATASETS.ROOT_DIR
        # Read the merged parcel data geo json file, This contains the parcel id and the geometry of all the parcels
        self.data = gpd.read_file(self.path + "parcel_data.geojson")
        # Get the image paths and corresponding pixel mask arrays
        self.data = self._get_image_pixel_masks(parcel_data=self.data, size=(224, 224))     #! Specify the image size which the model can expect
        
        self.image_size = None
        self.scale_factor = 1.0 # By default the scale factor is 1.0, which means the image is not scaled
        
        self.length = len(self.data)

        self.model = prepare_sam(checkpoint=cfg.MODEL.CHECKPOINT, model_type = 'base')

        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        This function returns one sample from the dataset.

        Args: idx (list): List of indexes.

        Returns:
            Image as data
            List of pixel masks
            Image size (height, width)
            Scale factor (Float) 
        """

        # Read the image from the path
        image_path = self.data[idx][0]    
        image = cv2.imread(image_path)
        
        self.image_size = image.shape[:2]
        scale_factor = 1024/max(self.image_size)   #! This is the scale factor used to scale the image to 1024x1024
        gt_masks = self.data[idx][1]

        # # These metadata are used for the model
        # data = {
        #     "image": image,                     #The image will be scaled before being passed to the model in the engine
        #     "gt_masks": gt_masks,               #The masks will be scaled before being passed to the model in the engine
        #     "image_size": self.image_size,      #This is the original image size
        #     "scale_factor": self.scale_factor   #This is the scale factor used to scale the image to 1024x1024
        # }
        # return data
        return {
            "image": image,
            "scale_factor": scale_factor,
            "gt_masks": gt_masks,
            "image_size": self.image_size, 
            "parcel_id": image_path
        }

    def _convert_polygons_to_pixels(self, parcel_geometry, polygon_labels, size):
        # Create a black background
        background = np.zeros(size, dtype=np.uint8)
         
        # Get the polygon bounds
        min_lon, min_lat, max_lon, max_lat = parcel_geometry.bounds
        
        # Get the width and height of the parcel
        height = max_lat - min_lat
        width = max_lon - min_lon

        # Get the pixel width and height
        pixel_height = size[0]/height
        pixel_width = size[1]/width

        pixel_masks = []
        for label in polygon_labels.geometry:
            pixel_mask = background.copy()
            # Iterate over the polygons in the multipolygon
            for polygon in label.geoms:
                coords = polygon.exterior.coords.xy

                # Get the pixel coords
                pixel_coords = []
                for i in range(len(coords[0])):
                    x = coords[0][i]
                    y = coords[1][i]

                    # Get the pixel x and y
                    pixel_x = (x - min_lon) * pixel_width
                    pixel_y = (max_lat - y) * pixel_height

                    pixel_coords.append([pixel_x, pixel_y])

                # Convert to int to make it work with cv2
                pixel_coords = np.array(pixel_coords, dtype=np.int32)
                # Fill the polygons and append to the pixel mask
                cv2.fillPoly(pixel_mask, pts=[pixel_coords], color=255)
            pixel_masks.append(pixel_mask.astype(dtype=bool))
        
        # Return the pixel masks as a numpy array
        return pixel_masks    

    def _get_image_pixel_masks(self, parcel_data, size):
        """Convert the polygon masks to pixel masks
        Note:
            Refrain from sending image sizes more than 1024x1024. This can cause some issues with SAM image encoder.
            The engine has is not tested for images more than 1024x1024.
        Args:
            parcel_data (geopandas.GeoDataFrame): GeoDataFrame containing the parcel data.

        Returns:
            image_list: List of image paths.
            list: List of pixel masks.
        """
        
        # Store the processed data in a list
        processed_data = []

        console = Console()

        with console.status("[bold green]Loading data...", spinner="earth") as status:
            for _, data in parcel_data.iterrows():
                parcel_id = data['parcel_id']
                geometry = data['geometry']
                
                polygon_mask_list = gpd.read_file(self.path + parcel_id + '.geojson')
                
                # If the parcel has no labels, let's keep it with no labels and no prompts
                if len(polygon_mask_list) == 0:
                    # Store the image path based on the current geoJSON file name only if the parcel has labels
                    processed_data.append((self.path + parcel_id + '.png', []))
                else:
                    # Get the pixel masks for the parcel
                    pixel_masks = self._convert_polygons_to_pixels(geometry, polygon_mask_list, size)
                    # Get the bounding boxes and point prompts for the parcel
                    processed_data.append((self.path + parcel_id + '.png', pixel_masks))

                status.update(f"[bold green]Loading data... Found [blue]{len(processed_data)} [green]Images")

        console.log(f"[bold green]Data loaded! Total number of images: [blue]{len(processed_data)}")

        return processed_data
    

if __name__ == "__main__":
    dataset = ParcelDataset(cfg)