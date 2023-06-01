# encoding: utf-8
"""
@author:  Aninda Ghosh
@contact: aghosh57@asu.edu
"""

import numpy as np
import cv2
import geopandas as gpd
from torch.utils.data import Dataset, DataLoader

class ParcelDataset(Dataset):
    """Parcel dataset
    --------------------------------
    This class is used to load the dataset.
    The data is stored in a folder with the following structure:
    - data
        - france
            - parcels
                - xxxxx.png
                - xxxxx.geojson
                - ...

    The labels is loaded using the `geopandas` library.
    The images are loaded using the `matplotlib.pyplot` module.
    The labels are stored in boolean masks. The masks are stored in a list.
    """
    
    def __init__(self, path, size=(224, 224)):
        """
        Args:
            path (string): Path to the folder containing the dataset.
        """
        self.path = path

        # TODO: I don't know what the file structure is going to be like, NEED Discussion with Manthan
        self.data = gpd.read_file(path + "parcel_data.geojson") 
        self.image_paths, self.bbox_prompts, self.point_prompts, self.labels = self._get_image_prompts_labels(self.data, size=size)
        self.length = len(self.data)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        This function returns the image and the pixel mask for given list of indexes.

        Args: idx (list): List of indexes.

        Returns:
            zipped
            np.array: Image as data
            np.array: List of pixel masks.
            np.array: List of box prompts.
            np.array: List of point prompts.
        """
        # TODO: Need to fix this function to return single image, bbox prompt, point prompt and pixel mask            
        image_path = self.images[idx]
        label = self.labels[idx]
        
        # Handle the case where the idx is not a list
        if isinstance(image_path, str):
            image_path = [image_path]
                
        all_images = []
        for path in image_path:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            all_images.append(image)
        
        # Zipping the images and the pixel masks for random shuffling in the data loader
        return zip(np.array(all_images), np.array(label, dtype=object))

    def _convert_polygons_pixel_maps(self, row, label_list, size):
        # Create a black background
        background = np.zeros(size, dtype=np.uint8)
        
        # Get the polygon bounds
        min_lon, min_lat, max_lon, max_lat = row.geometry.bounds
        
        # Get the width and height of the parcel
        height = max_lat - min_lat
        width = max_lon - min_lon

        # Get the pixel width and height
        pixel_height = size[0]/height
        pixel_width = size[1]/width

        pixel_masks = []
        for label in label_list.geometry:
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
        return pixel_masks

    def _get_bbox_and_point_prompts(self, pixel_masks):
        # TODO: This function needs to be implemented to obtain the bbox and point prompts
        bbox_prompts = []
        point_prompts = []
        for mask in pixel_masks:
            # Get the bounding box
            pass

            # Get the point prompts
            pass

        return bbox_prompts, point_prompts

    def _get_image_prompts_labels(self, parcel_data, size = (448, 448)):
        """Convert the polygon masks to pixel masks

        Args:
            parcel_data (geopandas.GeoDataFrame): GeoDataFrame containing the parcel data.

        Returns:
            image_list: List of image paths.
            list: List of pixel masks.
        """
        
        # Store the processed data in a dictionary
        processed_data = {
            'image_paths': [],
            'pixel_masks': []
        }
        
        for _, row in parcel_data.iterrows():
            # Read the label list first to check if the parcel has labels
            label_list = gpd.read_file(self.path + row['parcel_id'] + '.geojson')

            # If the parcel has no labels, Ignore that parcel
            if len(label_list) == 0:
                continue
            
            # Store the image path based on the current geoJSON file name only if the parcel has labels
            processed_data['image_paths'].append(self.path + row['parcel_id'] + '.png')

            

            # Append the pixel masks to the list
            processed_data['pixel_masks'].append(np.array(pixel_masks))

        return processed_data['image_paths'], processed_data['bbox_prompts'], processed_data['point_prompts'], processed_data['pixel_masks']