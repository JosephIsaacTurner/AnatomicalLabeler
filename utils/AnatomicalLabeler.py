import nibabel as nib
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
tqdm.pandas() 

# Get the directory in which the current script is located
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

# Construct the path to the 'atlases' directory
atlases_dir = os.path.join(parent_dir, 'atlas_data')

class Atlas:
    """
    A class for handling and accessing data from neuroimaging atlases.

    This class is designed to load and manage an atlas used in neuroimaging, typically consisting of
    a Nifti file (.nii or .nii.gz) representing brain regions and an accompanying CSV file that maps
    numerical values in the Nifti file to named regions.

    Attributes:
        csv_key_path (str): Path to the CSV file containing the key for the atlas.
        key (pd.DataFrame): DataFrame loaded from the CSV file. It maps values in the Nifti file to named regions.
        atlas (nib.Nifti1Image): Nifti image loaded from the Nifti file.
        labels (list): List of region names extracted from the CSV key.
        roi_size: Size of the region of interest (ROI) in the atlas.

    Args:
        filepath (str): Path to the .nii/.nii.gz file of the atlas or to the corresponding .csv file.
        csv_key_path (str, optional): Optional path to the .csv file containing the key for the atlas.
                                      If not provided, the class attempts to find a .csv file matching
                                      the root name of the provided Nifti file.
    """

    def __init__(self, filepath=os.path.join(atlases_dir, "HarvardOxford-cort-maxprob-thr0-2mm.nii.gz"), csv_key_path=None):
        """
        Initializes the Atlas class by loading the atlas Nifti file and the corresponding CSV key.
        """
        
        # Determine the root file name without extension
        root_filepath = os.path.splitext(filepath)[0]
        if root_filepath.endswith('.nii'):
            root_filepath = os.path.splitext(root_filepath)[0]

        # Set up file paths for .csv and .nii/.nii.gz files
        if not csv_key_path:
            csv_key_path = root_filepath + ".csv"

        nii_file_path = root_filepath + ".nii.gz"
        if not os.path.exists(nii_file_path):
            nii_file_path = root_filepath + ".nii"
            if not os.path.exists(nii_file_path):
                raise FileNotFoundError(f"Could not find a .nii or .nii.gz file for {root_filepath}.")

        if not os.path.exists(csv_key_path):
            raise FileNotFoundError(f"Could not find the csv file {csv_key_path}.")

        # Load the atlas and the key
        self.csv_key_path = csv_key_path
        self.key = pd.read_csv(self.csv_key_path)
        self.atlas = nib.load(nii_file_path)
        self.labels = self.key['name'].tolist()
    
    def name_at_index(self, index=[48, 94, 35]):
        """
        Retrieves the name of the brain region corresponding to a given index in the atlas.

        This method looks up the value at the specified index in the Nifti file and uses the CSV key
        to return the name of the region associated with that value.

        Args:
            index (list): A list of three integers representing the x, y, z coordinates of the index (in voxel space).

        Returns:
            str or list: The name of the region at the given index. If multiple regions are found at the index,
                         returns a list of names. Returns "No matching region found" if no region matches the index.
        """
        
        value_at_index = self.atlas.get_fdata()[tuple(index)]
        matched_row = self.key[self.key['value'] == value_at_index]
        if len(matched_row) == 1:
            return matched_row['name'].iloc[0]
        elif len(matched_row) > 1:
            return matched_row['name'].tolist()
        else:
            return "No matching region found"
        
    def roi_size(self, roi_name):
        """
        Returns the size of a region of interest (ROI) in the atlas.

        This method calculates the size of a region of interest (ROI) in the atlas by summing the
        number of voxels in the Nifti image that correspond to the specified region.

        Args:
            roi_name (str): The name of the region of interest (ROI) for which to calculate the size.

        Returns:
            int: The size of the region of interest (ROI) in the atlas.
        """
        if roi_name not in self.key['name'].values:
            return None
        roi_mask = self.key[self.key['name'] == roi_name]['value'].values[0]
        return np.sum(self.atlas.get_fdata() == roi_mask)

class AtlasLabeler:
    """
    A class to label Nifti volumes based on an atlas.

    This class is designed to take a Nifti volume and an atlas, and label the volume
    based on the regions defined in the atlas. The labeling considers the intensity
    thresholds specified for the Nifti volume.

    Attributes:
        atlas (Atlas): An Atlas object containing the atlas data and methods.
        labels (list): List of region names from the atlas.
        nifti (nib.nifti1.Nifti1Image): The Nifti image to be labeled.
        volume_data (numpy.ndarray): Data from the Nifti image.
        labeled_data (pd.DataFrame or None): DataFrame containing labeled voxels. Populated after label_volume is called.
        voxel_counts (dict or None): Dictionary of counts of labeled voxels per region. Populated after label_volume is called.
        unique_labels (list or None): List of unique labels assigned. Populated after label_volume is called.
        proportion_of_volume (dict or None): Proportion of the volume occupied by each label. Populated after label_volume is called.
        proportion_of_anatomical_label (dict or None): Proportion of each laebl (anatomical region) occupied by the volume. Populated after label_volume is called.
        min_threshold (int): Minimum intensity threshold for considering a voxel in the Nifti volume.
        max_threshold (int): Maximum intensity threshold for considering a voxel in the Nifti volume.

    Args:
        nifti (str or nib.nifti1.Nifti1Image): The Nifti volume to be labeled, either as a file path or Nifti image object.
        atlas (Atlas or str): An Atlas object or the file path to the atlas (.nii.gz file).
        min_threshold (int, optional): Minimum intensity value to consider for labeling. Defaults to 1.
        max_threshold (int, optional): Maximum intensity value to consider for labeling. Defaults to 1.
        total_voxel_count (int): Total number of voxels in the volume between the min and max thresholds.
    """

    def __init__(self, nifti, atlas=os.path.join(atlases_dir,"HarvardOxford-cort-maxprob-thr0-2mm.nii.gz"), min_threshold=1, max_threshold=1):
        """
        Initializes the AtlasLabeler class with a Nifti volume, an atlas, and intensity thresholds.
        """

        if type(atlas) != Atlas:
            # print(f"Looking for atlas {atlas}...")
            try:
                atlas = Atlas(atlas)
            except:
                print(f"""Could not find atlas {atlas}. Make sure the path is correct and try again. 
                      \n Example: atlases/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz""")
                return
        self.atlas = atlas
        self.labels = atlas.labels

        if type(nifti) != nib.nifti1.Nifti1Image:
            # print(f"Looking for nifti {nifti}...")
            try:
                nifti = nib.load(nifti)
            except:
                print(f"""Could not find nifti {nifti}. Make sure the path is correct and try again. 
                      \n Example: volumes/subject_lesion_mask.nii.gz""")
                return
        self.nifti = nifti
        self.volume_data = nifti.get_fdata()
        self.unique_labels = None
        self.labeled_data = None
        self.voxel_counts = None

        self.proportion_of_volume = None
        self.proportion_of_anatomical_label = None

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
    
    @property
    def total_voxel_count(self):
        """Returns the total number of voxels in volume between the min and max thresholds."""
        return np.sum((self.volume_data >= self.min_threshold) & (self.volume_data <= self.max_threshold))

    def label_volume(self):
        """
        Labels the Nifti volume using the regions defined in the atlas.

        This method iterates through each voxel in the Nifti volume, labeling it based on the
        overlapping regions in the atlas masks. Only voxels with intensity values within the
        specified threshold range are considered for labeling.

        The method updates the labeled_data, voxel_counts, and unique_labels attributes of the
        class with the results of the labeling process.

        Returns:
            AtlasLabeler: The instance itself with updated labeled_data, voxel_counts, and unique_labels.
        """        
        if self.atlas.atlas.shape != self.nifti.shape:
            print(f"The shape of the atlas ({self.atlas.atlas.shape}) and the nifti volume ({({self.nifti.shape})}) do not match. Please provide a nifti volume with the same shape as the atlas.")
            return

        # Find indices where the volume data is within the specified threshold range
        masked_indices = np.where(np.logical_and(self.volume_data >= self.min_threshold, 
                                         self.volume_data <= self.max_threshold))
        # Prepare a dictionary to store the results
        results = {'index': [], 'atlas_label': []}

        # Iterate over the masked indices
        for i, j, k in zip(*masked_indices):
            label = self.atlas.name_at_index([i, j, k])

            # Store the results
            results['index'].append((i, j, k))
            results['atlas_label'].append(label)

        # Convert the results to a DataFrame
        results_df = pd.DataFrame(results)
        self.labeled_data = results_df
        self.voxel_counts = results_df['atlas_label'].value_counts().to_dict()
        self.unique_labels = results_df['atlas_label'].unique().tolist()

        self.proportion_of_volume = {label: count / self.total_voxel_count for label, count in self.voxel_counts.items()}
        self.proportion_of_anatomical_label = {label: count / (self.atlas.roi_size(label) or 1) for label, count in self.voxel_counts.items()}
        return self

class MultiAtlasLabeler:
    """
    A class for labeling a Nifti volume using multiple atlases.

    This class enables the labeling of a Nifti volume by applying multiple atlas definitions
    with specified intensity thresholds. It consolidates the labeling results from all the
    atlases, providing comprehensive labeling based on multiple sources.

    Attributes:
        atlas_list (pd.DataFrame): A DataFrame containing paths to atlases and their corresponding loaded Atlas objects.
        labels (list): A list of unique labels across all atlases.
        nifti_obj (str or nib.Nifti1Image): The file path or Nifti image object of the volume to be labeled.
        voxel_counts (dict or None): Counts of labeled voxels per region after label_volume is called (aggregated across all atlases).
        unique_labels (list or None): Unique labels assigned to the volume after label_volume is called (aggregated across all atlases).
        min_threshold (int): Minimum intensity threshold for considering a voxel in the labeling process.
        max_threshold (int): Maximum intensity threshold for considering a voxel in the labeling process.
        total_voxel_count (int): Total number of voxels in the volume between the min and max thresholds.
        proportion_of_volume (dict or None): Proportion of the volume occupied by each label. Populated after label_volume is called.
        proportion_of_anatomical_label (dict or None): Proportion of each laebl (anatomical region) occupied by the volume. Populated after label_volume is called.

    Args:
        nifti_obj (str or nib.Nifti1Image): The file path or Nifti image object of the volume to be labeled.
        atlas_list (str or list or dict): Path to a CSV file containing atlas paths, a list of atlas paths, or a dictionary with atlas paths. The CSV file or dictionary should have a column or key named 'atlas_path'.
        min_threshold (int, optional): Minimum intensity value to consider for labeling. Defaults to 1.
        max_threshold (int, optional): Maximum intensity value to consider for labeling. Defaults to 1.
    """  
    def __init__(self, nifti_obj, atlas_list=os.path.join(atlases_dir,'harvoxf_atlas_list.csv'), min_threshold=1, max_threshold=1):
        # Store the thresholds
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # Load atlas_list from a CSV file
        if isinstance(atlas_list, str):
            try:
                atlas_list = pd.read_csv(atlas_list)
                atlas_list['atlas'] = atlas_list['atlas_path'].apply(lambda x: Atlas(x))
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find csv {atlas_list}. Make sure the path is correct and try again.")

        # Handling atlas_list if it's a list or a dict
        elif isinstance(atlas_list, (list, dict)):
            processed_atlas_list = [Atlas(item) if not isinstance(item, Atlas) else item for item in atlas_list]
            atlas_list = pd.DataFrame({'atlas': processed_atlas_list})

        else:
            raise ValueError("Invalid type for atlas_list. Must be a string, list, or dictionary.")

        # Load Nifti file
        if not isinstance(nifti_obj, nib.nifti1.Nifti1Image):
            try:
                self.nifti_obj = nib.load(nifti_obj)
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find nifti {nifti_obj}. Make sure the path is correct and try again.")
        else:
            self.nifti_obj = nifti_obj

        consolidated_unique_labels = []
        for atlas in atlas_list['atlas']:
            labels = atlas.labels
            consolidated_unique_labels.extend(labels)
        
        self.labels = list(set(consolidated_unique_labels))

        self.atlas_list = atlas_list
        self.voxel_counts = None
        self.unique_labels = None
    
    @property
    def total_voxel_count(self):
        """Returns the total number of voxels in volume between the min and max thresholds."""
        return np.sum((self.nifti_obj.get_fdata() >= self.min_threshold) & (self.nifti_obj.get_fdata() <= self.max_threshold))

    def label_volume(self):
        """
        Labels the Nifti volume using the atlases provided in the atlas list.

        This method applies each atlas in the atlas list to label the Nifti volume. It then consolidates
        the labeling results from all the atlases, providing a comprehensive set of labels for the volume.

        The method updates the voxel_counts and unique_labels attributes of the class with the consolidated results.

        Returns:
            MultiAtlasLabeler: The instance itself with updated voxel_counts and unique_labels.
        """

        def safe_label_volume(nifti_obj, atlas, min_threshold, max_threshold):
            try:
                labeler = AtlasLabeler(nifti_obj, atlas, min_threshold, max_threshold)
                labeler.label_volume()
                return labeler
            except Exception as e:
                print(f"Error processing {atlas}: {e}")
                return None
            
        def find_roi_size(label):
            for _, row in self.atlas_list.iterrows():
                if label in row['atlas'].labels:
                    return row['atlas'].roi_size(label)
            return None

        # List to hold the results for each atlas
        results_list = []

        # Iterate through the atlas list and label the volume for each atlas
        for _, row in self.atlas_list.iterrows():
            labeler = safe_label_volume(self.nifti_obj, row['atlas'], self.min_threshold, self.max_threshold)
            if labeler and labeler.labeled_data is not None:
                voxel_counts = labeler.voxel_counts
                unique_labels = labeler.unique_labels
                results_list.append({'voxel_counts': voxel_counts, 'unique_labels': unique_labels})

        # Convert the list of results to a DataFrame using pd.concat
        if results_list:
            # Aggregate voxel counts
            combined_voxel_counts = {}
            for result in results_list:
                for label, count in result['voxel_counts'].items():
                    combined_voxel_counts[label] = combined_voxel_counts.get(label, 0) + count
            self.voxel_counts = combined_voxel_counts

            # Aggregate unique labels
            all_labels = set()
            for result in results_list:
                all_labels.update(result['unique_labels'])
            self.unique_labels = list(all_labels)
        else:
            print("No valid results were generated from the atlases.")
            self.voxel_counts = {}
            self.unique_labels = []

        self.proportion_of_volume = {label: count / self.total_voxel_count for label, count in self.voxel_counts.items()}
        self.proportion_of_anatomical_label = {label: count / (find_roi_size(label) or 1) for label, count in self.voxel_counts.items()}

        return self

class CustomAtlas:
    """
    A class for managing an atlas used in neuroimaging analysis.

    This class handles the loading and management of an atlas, which includes a set
    of regions (each with a corresponding mask) used for analyzing or labeling Nifti volumes.
    
    Attributes:
        atlas_df (pd.DataFrame): A DataFrame representing the atlas. Each row corresponds to a region, with columns for region name and mask file path.
        labels (list): A list of region names present in the atlas.

    Args:
        atlas_path (str or dict or pd.DataFrame): The path to the CSV file representing the atlas, 
                                                  a dictionary, or a DataFrame with atlas information.
                                                  The CSV or DataFrame should have columns for 'region_name' 
                                                  and 'mask_path'. If using a dictionary, the keys should be
                                                  'region_name' and 'mask_path'.
    """

    def __init__(self, atlas_path=os.path.join(atlases_dir, "joseph_custom_atlas.csv")):
        if type(atlas_path) == str:
            try:
                self.atlas_df = pd.read_csv(atlas_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find csv {atlas_path}. Make sure the path is correct and try again.")
        elif isinstance(atlas_path, (dict, pd.DataFrame)):
            self.atlas_df = pd.DataFrame(atlas_path) if isinstance(atlas_path, dict) else atlas_path
        else:
            raise ValueError("Invalid atlas_path type. Please provide a string path, a dictionary, or a DataFrame.")
        
        self.labels = self.atlas_df['region_name'].tolist()
    
    def name_at_index(self, index=[48, 94, 35]):
        """Returns the names of the regions at the given index.
        Args:
            index (list): List of three integers representing the x, y, z coordinates of the index (in voxel space).
        Returns:
            list: Names of the regions at the given index."""
        
        region_names = []
        for _, row in self.atlas_df.iterrows():
            mask_path = row['mask_path']
            mask_volume = nib.load(mask_path).get_fdata()
            if mask_volume[tuple(index)] > 0:
                region_names.append(row['region_name'])

        if len(region_names) == 0:
            return ["No matching region found"]
        else:
            return region_names

    def roi_size(self, roi_name):
        """Returns the size of a region of interest (ROI) in the atlas.
        Args:
            roi_name (str): The name of the region of interest (ROI) for which to calculate the size.
        Returns:
            int: The size of the region of interest (ROI) in the atlas."""
        if roi_name not in self.atlas_df['region_name'].values:
            return None
        mask_path = self.atlas_df[self.atlas_df['region_name'] == roi_name]['mask_path'].values[0]
        mask_volume = nib.load(mask_path).get_fdata()
        return np.sum(mask_volume > 0)
    
class CustomAtlasLabeler:
    """
    A class for labeling Nifti volumes based on predefined regions from an atlas.

    This class takes a Nifti volume and an atlas (CustomAtlas) and labels the volume
    based on the regions defined in the atlas. The labeling is done based on the 
    intensity thresholds specified for the Nifti volume.

    Attributes:
        atlas (CustomAtlas): An instance of the CustomAtlas class.
        labels (list): List of region names from the atlas.
        atlas_df (pd.DataFrame): DataFrame representing the atlas, with region names and mask paths.
        nifti_obj(nib.Nifti1Image): The nifti_objimage to be labeled.
        volume_data (numpy.ndarray): Data from the nifti_objimage.
        min_threshold (int): Minimum intensity threshold for considering a voxel in the nifti_objvolume.
        max_threshold (int): Maximum intensity threshold for considering a voxel in the nifti_objvolume.
        labeled_data (pd.DataFrame or None): DataFrame containing labeled voxels after label_volume is called.
        voxel_counts (dict or None): Dictionary containing counts of labeled voxels per region after label_volume is called.
        unique_labels (list or None): List of unique labels assigned after label_volume is called.
        total_voxel_count (int): Total number of voxels in the volume between the min and max thresholds.

    Args:
        nifti_obj (str or nib.Nifti1Image): File path to the nifti_obj volume or a nifti_obj image object.
        atlas (CustomAtlas or str (path to atlas)): An instance of CustomAtlas or a file path to an atlas CSV file.
        min_threshold (int, optional): Minimum intensity value to consider for labeling. Default is 1.
        max_threshold (int, optional): Maximum intensity value to consider for labeling. Default is 1.
    """

    def __init__(self, nifti_obj, atlas=os.path.join(atlases_dir, "joseph_custom_atlas.csv"), min_threshold=1, max_threshold=1):
        
        if not isinstance(atlas, CustomAtlas):
            atlas = CustomAtlas(atlas)
        
        if not isinstance(nifti_obj, nib.nifti1.Nifti1Image):
            try:
                nifti_obj = nib.load(nifti_obj)
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find nifti_obj {nifti_obj}. Make sure the path is correct and try again.")

        self.atlas = atlas
        self.labels = self.atlas.labels
        self.atlas_df = atlas.atlas_df
        self.nifti_obj = nifti_obj
        self.volume_data = nifti_obj.get_fdata()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.labeled_data = None
        self.voxel_counts = None
        self.unique_labels = None 
        self.proportion_of_volume = None
        self.proportion_of_anatomical_label = None
    
    @property
    def total_voxel_count(self):
        """Returns the total number of voxels in volume between the min and max thresholds."""
        return np.sum((self.volume_data >= self.min_threshold) & (self.volume_data <= self.max_threshold))

    def label_volume(self):
        """
        Labels the Nifti volume using the regions defined in the atlas.

        This method iterates through each voxel in the Nifti volume and labels it based
        on the overlapping regions in the atlas masks. Only voxels with intensity values
        within the specified threshold range are considered for labeling.

        The method updates the labeled_data, voxel_counts, and unique_labels attributes
        of the class with the results of the labeling process.

        Returns:
            CustomAtlasLabeler: The instance (itself) with updated labeled_data, voxel_counts, and unique_labels.
        """        
        
        # Initialize a dictionary to hold the labeling results
        results = {'index': [], 'atlas_label': []}
        
        # Identify voxels in the nifti volume that meet the threshold criteria
        within_threshold_voxels = set(zip(*np.where((self.volume_data >= self.min_threshold) & 
                                                    (self.volume_data <= self.max_threshold))))

        # Define a function to process each row of the DataFrame
        def process_row(row):
            region_name, mask_path = row['region_name'], row['mask_path']
            # Load the mask data
            mask_volume = nib.load(mask_path).get_fdata()
            # Find voxels in the mask that are non-zero
            mask_active_voxels = set(zip(*np.where(mask_volume > 0)))

            # Find intersection of within-threshold voxels and mask-active voxels
            intersecting_voxels = within_threshold_voxels.intersection(mask_active_voxels)
            for voxel_coords in intersecting_voxels:
                # Save the voxel coordinates and the corresponding label (region name) in the results
                results['index'].append(voxel_coords)
                results['atlas_label'].append(region_name)

        # Apply the function to each row of the DataFrame
        self.atlas_df.apply(process_row, axis=1)

        # Transform the collected results into a DataFrame
        results_df = pd.DataFrame(results)
        self.labeled_data = results_df
        self.voxel_counts = results_df['atlas_label'].value_counts().to_dict()
        self.unique_labels = results_df['atlas_label'].unique().tolist()

        self.proportion_of_volume = {label: count / self.total_voxel_count for label, count in self.voxel_counts.items()}
        self.proportion_of_anatomical_label = {label: count / (self.atlas.roi_size(label) or 1) for label, count in self.voxel_counts.items()}

        return self

def determine_predominant_label(row, col_name_option):
    attribute_value = getattr(row['labeling_results'], col_name_option)
    
    # Filter out 'No matching region found' from the dictionary
    filtered_attribute_value = {k: v for k, v in attribute_value.items() if k not in ['No matching region found', 'cerebral_cortex', 'subcortex']}
    
    # Check if filtered_attribute_value is not empty to avoid ValueError from max()
    if filtered_attribute_value:
        # Find the label with the maximum value from the filtered dictionary
        predominant_label = max(filtered_attribute_value.items(), key=lambda x: x[1])[0]
    
    else:
        predominant_label = 'No matching region found'
    
    return predominant_label

def add_predominant_label(df, col_name_option='voxel_counts'):
    df['predominant_label'] = df.apply(determine_predominant_label, col_name_option=col_name_option, axis=1)
    return df

def label_csv_atlas(csv_path, 
                    roi_col_name="roi_2mm", 
                    min_threshold=1, 
                    max_threshold=1, 
                    proportion_of_volume=False,
                    proportion_of_anatomical_label=False,
                    atlas=os.path.join(atlases_dir,"HarvardOxford-cort-maxprob-thr0-2mm.nii.gz")):
    """
    Labels a set of Nifti volumes specified in a CSV file or DataFrame using a provided atlas.

    This function processes either a CSV file or a DataFrame where each row corresponds to a Nifti volume.
    It labels each volume using the specified atlas and aggregates voxel counts for each region in the atlas.
    These counts are appended as new columns to the DataFrame.

    Args:
        csv_path (str or DataFrame): Path to the CSV file or a DataFrame containing information about the Nifti volumes.
                                     The CSV file or DataFrame is expected to have a column 'orig_roi_vol' containing
                                     paths to the Nifti volumes.
        roi_col_name (str, optional): Name of the column in the CSV file containing the paths to the Nifti volumes.
                                        Defaults to "orig_roi_vol".
        min_threshold (int, optional): Minimum intensity value to consider for labeling. Defaults to 1.
        max_threshold (int, optional): Maximum intensity value to consider for labeling. Defaults to 1.
        proportion_of_volume (bool, optional): Whether to calculate the proportion of the volume occupied by each label.
                        Defaults to False.
        proportion_of_anatomical_label (bool, optional): Whether to calculate the proportion of each label (anatomical region)
        atlas (Atlas or str): An instance of the Atlas class or a string path to the atlas file (.nii, .nii.gz, or .csv)
                              to be used for labeling the volumes.
    
    Returns:
        pd.DataFrame: A DataFrame with the original data from the CSV file or DataFrame, augmented with voxel counts for
                      each region in the atlas for each Nifti volume. The voxel counts are added as new columns corresponding
                      to each region in the atlas.
    """
    if type(csv_path) == str:
        df = pd.read_csv(csv_path)
    elif type(csv_path) == pd.DataFrame:
        df = csv_path
    else:
        raise ValueError(f"csv_path must be a string or a DataFrame. {type(csv_path)} was provided.")
    df['labeling_results'] = df[roi_col_name].progress_apply(lambda x: AtlasLabeler(x, atlas, min_threshold, max_threshold).label_volume())
    df['total_voxel_count'] = df['labeling_results'].apply(lambda x: x.total_voxel_count)
    
    if proportion_of_volume:
        col_name = 'proportion_of_volume'
        df[col_name] = df['labeling_results'].apply(lambda x: x.proportion_of_volume)
    elif proportion_of_anatomical_label:
        col_name = 'proportion_of_anatomical_label'
        df[col_name] = df['labeling_results'].apply(lambda x: x.proportion_of_anatomical_label)
    else:
        col_name = 'voxel_counts'
        df[col_name] = df['labeling_results'].apply(lambda x: x.voxel_counts)
    
    df = add_predominant_label(df)

    labels = df['labeling_results'].iloc[0].labels
    for label in labels:
        df[label] = df[col_name].apply(lambda x: x.get(label, 0))
    df.drop(columns=[col_name, 'labeling_results'], inplace=True)

    # Drop columns where the entire column's values are 0
    df = df.loc[:, (df != 0).any(axis=0)]
    return df

def label_csv_multi_atlas(csv_path, 
                          roi_col_name="roi_2mm", 
                          min_threshold=1, 
                          max_threshold=1, 
                          proportion_of_volume=False,
                          proportion_of_anatomical_label=False,
                          atlas=[os.path.join(atlases_dir,"HarvardOxford-cort-maxprob-thr0-2mm.nii.gz"), os.path.join(atlases_dir,"HarvardOxford-sub-maxprob-thr0-2mm.nii.gz")]):
    """
    Labels a set of Nifti volumes specified in a CSV file or DataFrame using multiple provided atlases.

    This function processes either a CSV file or a DataFrame where each row corresponds to a Nifti volume.
    It applies labeling to each volume using multiple atlases specified in the atlases list and aggregates voxel 
    counts for each region in these atlases. The aggregated counts are then appended as new columns to the DataFrame.

    Args:
        csv_path (str or DataFrame): Path to the CSV file or a DataFrame containing information about the Nifti volumes.
                                     The CSV file or DataFrame is expected to have a column 'orig_roi_vol' containing
                                     paths to the Nifti volumes.
        roi_col_name (str, optional): Name of the column in the CSV file containing the paths to the Nifti volumes.
                                        Defaults to "orig_roi_vol".
        min_threshold (int, optional): Minimum intensity value to consider for labeling. Defaults to 1.
        max_threshold (int, optional): Maximum intensity value to consider for labeling. Defaults to 1.
        proportion_of_volume (bool, optional): Whether to calculate the proportion of the volume occupied by each label.
                        Defaults to False.
        proportion_of_anatomical_label (bool, optional): Whether to calculate the proportion of each label (anatomical region)
        atlas (list of Atlas or str): A list of Atlas instances or string paths to atlas files (.nii, .nii.gz, or .csv)
                                        to be used for labeling the volumes.
    
    Returns:
        pd.DataFrame: A DataFrame with the original data from the CSV file or DataFrame, augmented with voxel counts for
                      each region in all the atlases for each Nifti volume. The voxel counts are added as new columns corresponding
                      to each region in the atlases.
    """
    if type(csv_path) == str:
        df = pd.read_csv(csv_path)
    elif type(csv_path) == pd.DataFrame:
        df = csv_path
    else:
        raise ValueError(f"csv_path must be a string or a DataFrame. {type(csv_path)} was provided.")
    df['labeling_results'] = df[roi_col_name].progress_apply(lambda x: MultiAtlasLabeler(x, atlas_list=atlas, min_threshold=min_threshold, max_threshold=max_threshold).label_volume())
    df['total_voxel_count'] = df['labeling_results'].apply(lambda x: x.total_voxel_count)

    if proportion_of_volume:
        col_name = 'proportion_of_volume'
        df[col_name] = df['labeling_results'].apply(lambda x: x.proportion_of_volume)
    elif proportion_of_anatomical_label:
        col_name = 'proportion_of_anatomical_label'
        df[col_name] = df['labeling_results'].apply(lambda x: x.proportion_of_anatomical_label)
    else:
        col_name = 'voxel_counts'
        df[col_name] = df['labeling_results'].apply(lambda x: x.voxel_counts)
    
    df = add_predominant_label(df)

    labels = df['labeling_results'].iloc[0].labels
    
    for label in labels:
        df[label] = df[col_name].apply(lambda x: x.get(label, 0))
    df.drop(columns=[col_name, 'labeling_results'], inplace=True)
    
    # Drop columns where the entire column's values are 0
    df = df.loc[:, (df != 0).any(axis=0)]
    return df

def label_csv_custom_atlas(csv_path, 
                           roi_col_name="roi_2mm", 
                           min_threshold=1, 
                           max_threshold=1, 
                           proportion_of_volume=False,
                           proportion_of_anatomical_label=False,
                           atlas=os.path.join(atlases_dir,"joseph_custom_atlas.csv")):
    """
    Labels a set of Nifti volumes specified in a CSV file using a provided custom atlas.

    This function reads a CSV file, where each row corresponds to a Nifti volume, and applies
    labeling using the specified atlas. It extracts voxel counts for each region in the atlas and
    appends these counts as new columns in the DataFrame.

    Args:
        csv_path (str or DataFrame): Path to the CSV file containing information about the Nifti volumes.
                        The CSV file is expected to have a column 'roi_2mm' containing
                        paths to the Nifti volumes.
                        Alternatively, a DataFrame with the same structure as the CSV file can be provided.
        roi_col_name (str, optional): Name of the column in the CSV file containing the paths to the Nifti volumes.
                        Defaults to "orig_roi_vol".
        min_threshold (int, optional): Minimum intensity value to consider for labeling. Defaults to 1.
        max_threshold (int, optional): Maximum intensity value to consider for labeling. Defaults to 1.
        proportion_of_volume (bool, optional): Whether to calculate the proportion of the volume occupied by each label.
                        Defaults to False.
        proportion_of_anatomical_label (bool, optional): Whether to calculate the proportion of each label (anatomical region)
        atlas (CustomAtlas or str): An instance of the CustomAtlas class to be used for labeling the volumes.
                        Alternatively, a string path to the csv file of the custom atlas can be provided.
                        This should be a CSV file with columns 'region_name' and 'mask_path'.
    
    Returns:
        pd.DataFrame: A DataFrame with the original data from the CSV file, augmented with 
                      voxel counts for each region in the atlas for each Nifti volume.
    """
    if type(csv_path) == str:
        df = pd.read_csv(csv_path)
    elif type(csv_path) == pd.DataFrame:
        df = csv_path
    else:
        raise ValueError(f"csv_path must be a string or a DataFrame. {type(csv_path)} was provided.")
    df['labeling_results'] = df[roi_col_name].progress_apply(lambda x: CustomAtlasLabeler(x, atlas, min_threshold, max_threshold).label_volume())
    df['total_voxel_count'] = df['labeling_results'].apply(lambda x: x.total_voxel_count)
    
    if proportion_of_volume:
        col_name = 'proportion_of_volume'
        df[col_name] = df['labeling_results'].apply(lambda x: x.proportion_of_volume)
    elif proportion_of_anatomical_label:
        col_name = 'proportion_of_anatomical_label'
        df[col_name] = df['labeling_results'].apply(lambda x: x.proportion_of_anatomical_label)
    else:
        col_name = 'voxel_counts'
        df[col_name] = df['labeling_results'].apply(lambda x: x.voxel_counts)

    df = add_predominant_label(df)

    labels = df['labeling_results'].iloc[0].labels
    for label in labels:
        df[label] = df[col_name].apply(lambda x: x.get(label, 0))
    df.drop(columns=[col_name, 'labeling_results'], inplace=True)
    # Drop columns where the entire column's values are 0
    df = df.loc[:, (df != 0).any(axis=0)]
    return df

def quantification_from_csv( csv_path, 
                               roi_col_name='roi_2mm', 
                               label_function=label_csv_custom_atlas, 
                               proportion_of_anatomical_label=True,
                               proportion_of_volume=False,
                               min_threshold=1.0, 
                               max_threshold=1.0,
                               atlas=None
                               ):
    """
    Function takes a CSV file of NIfTI filepaths and quantifies the volume occupied by anatomical labels in an atlas.

    Args:
        - csv_path (str): The file path to the CSV file containing the NIfTI filepaths. Required.
        - roi_col_name (str): The column name of the NIfTI filepaths to label. Defaults to "roi_2mm".
        - label_function (function): The function to use for labeling the volumes. Defaults to label_csv_custom_atlas (using my custom atlas).
        - proportion_of_anatomical_label (bool): If True, the proportion of each label occupied by the volume is calculated. Defaults to True.
        - proportion_of_volume (bool): If True, the proportion of the volume occupied by each label is instead calculated. Defaults to False.
        - min_threshold (float): The minimum threshold for the proportion of the volume occupied by each label. Defaults to 1.0.
        - max_threshold (float): The maximum threshold for the proportion of the volume occupied by each label. Defaults to 1.0.
        - atlas (str): The file path to the atlas CSV file. If not provided, defaults to whatever is specified in the label_function.
    
    Returns:
        - labeled_df (DataFrame): The original DataFrame with the labeled volumes and their statistics.
        - stats_df (DataFrame): The statistics of the labeled volumes.
        - non_zero_frequency_df (DataFrame): The frequency of non-zero values for each label.
        - predominant_frequency_df (DataFrame): The frequency of predominant values for each label.

    """
    
    if proportion_of_anatomical_label:
        proportion_of_volume = False
    
    if atlas:
        labeled_df = label_function(csv_path=csv_path, 
                                    roi_col_name=roi_col_name, 
                                    min_threshold=min_threshold, 
                                    max_threshold=max_threshold,
                                    atlas=atlas, 
                                    proportion_of_anatomical_label=proportion_of_anatomical_label,
                                    proportion_of_volume=proportion_of_volume
                                    )
    else:
        labeled_df = label_function(csv_path=csv_path, 
                                    roi_col_name=roi_col_name, 
                                    min_threshold=min_threshold, 
                                    max_threshold=max_threshold,
                                    proportion_of_anatomical_label=proportion_of_anatomical_label,
                                    proportion_of_volume=proportion_of_volume
                                    )
    
    # Read the original CSV to identify new columns in labeled_df
    original_csv = pd.read_csv(csv_path)

    # Identify the new label columns in labeled_df not present in original_csv
    new_label_columns = [col for col in labeled_df.columns if col not in original_csv.columns and col != 'total_voxel_count' and col != 'predominant_label']

    # Initialize a list to hold the median and 25-75th percentile data
    label_stats = []

    # Iterate over the new label columns to calculate the statistics
    for label in new_label_columns:
        median = labeled_df[label].median()
        percentile_25 = labeled_df[label].quantile(0.25)
        percentile_75 = labeled_df[label].quantile(0.75)
        std_dev = labeled_df[label].std()  # Standard deviation
        mean = labeled_df[label].mean()     # Mean
        min_val = labeled_df[label].min()   # Minimum
        max_val = labeled_df[label].max()   # Maximum
            
        # Append the statistics for this label to the list
        label_stats.append({
            'Label': label,
            'Median': median,
            '25th Percentile': percentile_25,
            '75th Percentile': percentile_75,
            'Standard Deviation': std_dev,
            'Mean': mean,
            'Min': min_val,
            'Max': max_val
        })

    # Convert the list of dictionaries into a DataFrame
    stats_df = pd.DataFrame(label_stats).sort_values(by='Median', ascending=False)


    # Initialize a list to hold the label and its non-zero frequency
    non_zero_label_frequencies = []

    # Iterate over the new label columns to calculate the frequency of non-zero values
    for label in new_label_columns:
        non_zero_count = (labeled_df[label] > 0).sum()  # Count non-zero entries for each label
        
        # Append the frequency data for this label to the list
        non_zero_label_frequencies.append({
            'Label': label,
            'Non-zero Frequency': non_zero_count/len(labeled_df)  # Calculate the frequency of non-zero values
        })

    # Convert the list of dictionaries into a DataFrame
    non_zero_frequency_df = pd.DataFrame(non_zero_label_frequencies).sort_values(by='Non-zero Frequency', ascending=False)

    # Initialize a list to hold the label and its non-zero frequency
    predominant_label_frequencies = []

    # Iterate over the new label columns to calculate the frequency of non-zero values
    for label in new_label_columns:
        predominant_count = (label == labeled_df['predominant_label']).sum()  # Count predominant entries for each label

        # Append the frequency data for this label to the list
        predominant_label_frequencies.append({
            'Label': label,
            'Predominant Frequency': predominant_count/len(labeled_df)  # Calculate the frequency of non-zero values
        })

    # Convert the list of dictionaries into a DataFrame
    predominant_frequency_df = pd.DataFrame(predominant_label_frequencies)
    predominant_frequency_df = predominant_frequency_df[~predominant_frequency_df['Label'].isin(['cerebral_cortex', 'subcortex'])].sort_values(by='Predominant Frequency', ascending=False)

    # Return the stats DataFrame
    return labeled_df, stats_df, non_zero_frequency_df, predominant_frequency_df
