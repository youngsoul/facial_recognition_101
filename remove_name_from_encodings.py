import pickle
import argparse
from pathlib import Path
from shutil import rmtree

"""
The encodings.pkl is a dictionary of the form:

{
  'encodings': [ ndarray(), ndarray ],
  'names': ['name1','name2']
}

    f = open(encodings_file, "wb")
    f.write(pickle.dumps(data))
    f.close()

"""
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=False, default='./images/dataset',
                    help="path to input dataset directory.  If there are multiple directories all subdirectories will be encoded")
    ap.add_argument("-e", "--encodings-file", required=False, default='./encodings/facial_encodings.pkl',
                    help="path to serialized pickle file of facial encodings.  If the file exists, new encodings will be added.  Otherwise the file will be created")
    ap.add_argument("-n", "--name-to-remove", required=True,
                    help="name of the person to remove from encodings file.  ")

    args = vars(ap.parse_args())

    new_data = {
        'encodings': [],
        'names': []
    }

    encodings_file = args['encodings_file']
    name_to_remove = args['name_to_remove']

    with open(encodings_file, mode="rb") as opened_file:
        results = pickle.load(opened_file)
        for i, name in enumerate(results['names']):
            if name != name_to_remove:
                new_data['encodings'].append(results['encodings'][i])
                new_data['names'].append(name)


    # write new full set of encodings
    f = open(encodings_file, "wb")
    f.write(pickle.dumps(new_data))
    f.close()

    dataset_to_remove = f"{args['dataset']}/{name_to_remove}"

    dataset_path = Path(dataset_to_remove)
    rmtree(dataset_path)

