import pandas as pd
import os


def create_data_csv(dicom_path:str, csv_path:str, root_folder, output_csv:str)-> None:
    """
    Create a csv file with
    :param dicom_path:
    :param csv_file:
    :return:
    """
    #
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['im_path','label']) # first row

        for chunk in pd.read_csv(csv_path, chunksize=1):
            nom_dossier = chunk.iloc[0]['nom_dossier']
            label = chunk.iloc[0]['label']
            dossier_path = os.path.join(root_folder, nom_dossier)

            # Vérifier si le dossier existe
            if not os.path.isdir(dossier_path):
                print(f"** WARNING ** Folder not found : {dossier_path}")
                continue

            for image in os.listdir(dossier_path):
                image_path = os.path.join(dossier_path, image)
                if os.path.isfile(image_path):  # Vérifier que c'est une image
                    writer.writerow([image_path, label])