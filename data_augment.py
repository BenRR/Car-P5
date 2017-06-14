import csv
import cv2
import os

# default data set folders and files
data_dir = os.getcwd()
train_root_folder = '{}/training'.format(data_dir)
positive_folder = 'positive'
negative_folder = 'negative'
train_positive_folder = '{}/{}'.format(train_root_folder, positive_folder)
train_negative_folder = '{}/{}'.format(train_root_folder, negative_folder)
positive_file = '{}/vehicle.csv'.format(train_root_folder)
negative_file = '{}/non_vehicle.csv'.format(train_root_folder)


# flip images to create more data set
def flip(input_file, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)

    img = cv2.imread(input_file)
    #print(input_file)
    cv2.imwrite(output_file, cv2.flip(img, 1))


def create_training_set(files, csv_file, file_path, flag):
    with open(csv_file, 'w') as target_file:
        writer = csv.writer(target_file)
        for training_file in files:
            #if not training_file.startswith('flip') and not training_file.startswith('.'):
            #    flip_file = '{}/flip_{}'.format(file_path, training_file)
            #    flip('{}/{}'.format(file_path, training_file), flip_file)
            #    writer.writerow([flip_file, flag])
            if not training_file.startswith('.'):
                writer.writerow(['{}/{}'.format(file_path, training_file), flag])


if __name__ == '__main__':
    # re-create data folders if not exist
    if os.path.exists(positive_file):
        os.remove(positive_file)
    if os.path.exists(negative_file):
        os.remove(negative_file)

    positive_set = os.listdir(train_positive_folder)
    negative_set = os.listdir(train_negative_folder)

    create_training_set(positive_set, positive_file, train_positive_folder, '1')
    create_training_set(negative_set, negative_file, train_negative_folder, '0')
