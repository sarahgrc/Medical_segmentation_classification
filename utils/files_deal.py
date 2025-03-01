import pandas as pd
import os
import random
import shutil


def create_data_csv(dicom_path: str, csv_path: str, root_folder, output_csv: str) -> None:
    """
    Create a csv file with
    :param dicom_path:
    :param csv_file:
    :return:
    """
    #
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['im_path', 'label'])  # first row

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


def move_random_files(src_root: str, root_new_folder: str, p=0.1) -> None:
    """
    Move p % files randomly selected
    :param src_root:
    :param root_new_folder:
    :param p:
    :return:
    """

    if not os.path.exists(root_new_folder):
        os.makedirs(root_new_folder)

    list_im = [f for f in os.listdir(src_root) if os.path.isfile(os.path.join(src_root, f))]
    num2move = int(len(list_im) * p)
    im2move = random.sample(list_im, num2move)

    for im_path in im2move:
        src_path = str(src_root + '/' + im_path)
        dest_path = str(root_new_folder + '/' + im_path)
        shutil.move(src_path, dest_path)
    print('** done **')


if __name__ == '__main__' :
    # create folder validation with 10% of the initial training set
    move_random_files(src_root='dataset/archive/train/glioma_tumor',
                      root_new_folder='dataset/archive/val/glioma_tumor')

    move_random_files(
        src_root='dataset/archive/train/meningioma_tumor',
        root_new_folder='dataset/val/meningioma_tumor')

    move_random_files(
        src_root='dataset/train/no_tumor',
        root_new_folder='dataset/val/no_tumor')

    move_random_files(
        src_root='dataset/train/pituitary_tumor',
        root_new_folder='dataset/val/pituitary_tumor')