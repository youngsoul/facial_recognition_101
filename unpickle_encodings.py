import pickle

"""
The encodings.pkl is a dictionary of the form:

{
  'encodings': [ ndarray(), ndarray ],
  'names': ['name1','name2']
}

"""
if __name__ == '__main__':
    with open("./encodings/test_encodings.pkl", mode="rb") as opened_file:
        results = pickle.load(opened_file)

        print(results)