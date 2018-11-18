import zipfile


def unzip(zip_file_path, target_dir_path):
    """
    extract data from zip
    Args:
        zip_file_path (Path): path to zip file
        target_dir_path (Path): path to directory, in which extracted files are saved
    """
    print("unzipping...")
    with zipfile.ZipFile(zip_file_path) as existing_zip:
        existing_zip.extractall(target_dir_path)
    print("unzipped!")
