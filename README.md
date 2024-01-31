## AnatomicalLabeler: A Python Package for Identifying Anatomical Regions Regions in NIfTI Files

See 00_LabelVolumes.ipynb for usable examples.

**1. Anatomically label a NIfTI volume using a standard atlas**

    import AnatomicalLabeler

    # Path to the atlas (if not provided, defaults to the deterministic Harvard Oxford cortical atlas)
    atlas_path = r"atlases/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz" 

    volume_to_label = r'volumes/30lechowiczglogowska2019.nii.gz' # Path to the volume to label

    # Label the volume and return voxel counts and the unique regions in the volume
    labeled_results = AnatomicalLabeler.AtlasLabeler(volume_to_label, atlas_path).label_volume()
    labeled_results.voxel_counts, labeled_results.unique_labels

**2. Anatomically label a NIfTI volume using multiple standard atlases**

    import AnatomicalLabeler

    # List of paths to the atlases (if not provided, defaults to the deterministic Harvard Oxford cortical and subcortical atlases)
    atlases = [r"atlases/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz", r"atlases/HarvardOxford-sub-maxprob-thr0-2mm.nii.gz"] 

    volume_to_label = r'volumes/30lechowiczglogowska2019.nii.gz' # Path to the volume to label

    results_cort_and_sub = AnatomicalLabeler.MultiAtlasLabeler(volume_to_label, atlases).label_volume()
    results_cort_and_sub.voxel_counts, results_cort_and_sub.unique_labels

**3. Anatomically label a NIfTI volume using a custom atlas**

Custom atlases consist of CSVs with the following columns:
- `region_name` : The name of the region
- `mask_path` : The path to the corresponding anatomical mask file

Example:
        
    import AnatomicalLabeler

    # Path to custom_atlas csv file (can also be a DataFrame object)
    custom_atlas = r"atlases/joseph_custom_atlas.csv" # If not provided, defaults to my custom atlas (see the joseph_custom_atlas.csv file)

    volume_to_label = r'volumes/30lechowiczglogowska2019.nii.gz' # Path to the volume to label

    results_custom = AnatomicalLabeler.CustomAtlasLabeler(volume_to_label, custom_atlas).label_volume()
    results_custom.voxel_counts, results_custom.unique_labels

**4. Add anatomical voxel counts to a CSV or DataFrame of NIfTI files using standard, multiple, or custom atlases**

    from AnatomicalLabeler import label_csv_atlas, label_csv_multi_atlas, label_csv_custom_atlas

    # Path to custom_atlas csv file (can also be a DataFrame object)
    custom_atlas = r"atlases/joseph_custom_atlas.csv" # If not provided, defaults to my custom atlas (see the joseph_custom_atlas.csv file)
    csv_path = r"takotsubo_small.csv" # CSV file of volumes
    roi_col_name = "orig_roi_vol" # Column name of the NIfTI filepaths to label (defaults to "orig_roi_vol")

    # Replace function with label_csv_atlas or label_csv_multi_atlas for the other atlas types
    labeled_csv = label_csv_custom_atlas(csv_path, custom_atlas, roi_col_name)
    labeled_csv

**5. Advanced Options: Specify thresholds (for continuous volumes)**
- Useful when identifying regions of peak signal/significance in a continuous NIfTI volume

Example:

    from AnatomicalLabeler import label_csv_atlas, label_csv_multi_atlas, label_csv_custom_atlas

    # Path to custom_atlas csv file (can also be a DataFrame object)
    custom_atlas = r"atlases/joseph_custom_atlas.csv" # If not provided, defaults to my custom atlas (see the joseph_custom_atlas.csv file)
    csv_path = r"takotsubo_small.csv" # CSV file of volumes
    roi_col_name = "t" # Use t map instead of binary lesion mask

    min_threshold = 30
    max_threshold = 100

    # Replace function with label_csv_atlas or label_csv_multi_atlas for the other atlas types
    labeled_csv = label_csv_custom_atlas(csv_path, custom_atlas, roi_col_name, min_threshold, max_threshold)
    labeled_csv