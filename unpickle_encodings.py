import pickle

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
    with open("./encodings/facial_encodings.pkl", mode="rb") as opened_file:
        results = pickle.load(opened_file)

        print(list(set(results['names'])))