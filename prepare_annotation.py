import pickle

#truncate dictionary based on the index list
def truncate_dictionary(original_dict, index_lists):
    pass
    new_dictionary = {}
    index = 0
    for (key,value) in original_dict.items():
        if index in index_lists:
            new_dictionary[key]= original_dict[key]
        index += 1
    return new_dictionary


def read_dictionary(path):
    with open(path,'rb') as f:
        annotation_dict = pickle.load(f)
    return annotation_dict

def save_dicionary(path,annotation_dict):
    with open(path,'wb') as f:
        pickle.dump(annotation_dict, f)
        return 1


def preprocess_dictionary(path, set='train', save_path=None):
    # split the dataset into train, test and val
    if set == 'train':
        index_list = [i for i in range(0,140)]
    elif set == 'val':
        index_list =  [2,4,6,8,10,12,15,18,19,25,29,30,44,47,55,67,78,89,99,102,107,111,114,119,123,128,132,134,139,72]
    elif set == 'test':
        index_list = [0, 1, 3, 5, 7, 9, 11, 13, 14, 16, 17, 20, 21, 22, 23, 24, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 103, 104, 105, 106, 108, 109, 110, 112, 113, 115, 116, 117, 118, 120, 121, 122, 124, 125, 126, 127, 129, 130, 131, 133, 135, 136, 137, 138]

    annotation_dict = read_dictionary(path)
    annotation_dict = truncate_dictionary(annotation_dict,index_list)
    # print(annotation_dict)
    # print(len(annotation_dict))
    if save_path is None:
        save_path = path
    save_dicionary(save_path,annotation_dict)
    return 1

# append full path to the dictionary of image name
def append_dictionary(path_list, save_path):
    combine_dictionary = {}
    for path in path_list:
        dict_data = read_dictionary(path)
        combine_dictionary.update(dict_data)

    save_dicionary(save_path, combine_dictionary)
    # print(len(combine_dictionary))
    # print(combine_dictionary)
    return 1


# truncate dictionary

path_list = ['ssim_annotation/f1_160_ssim_annotation.pkl',
        'ssim_annotation/f2_145_ssim_annotation.pkl',
        'ssim_annotation/f5_153_ssim_annotation.pkl'
]

# path = path_list[2]

# path =  'hfen_annotation/f4_149_hfen_annotation.pkl'

# print(path)
# preprocess_dictionary(path,'train' )

# preprocess_dictionary(path,'test', 'hfen_annotation/test_hfen_annotation.pkl' ) # append full image path to dictionary


save_path = 'ssim_annotation/train_ssim_annotation.pkl'
append_dictionary(path_list,save_path) # append all dictionary from paths in path_list into single combined dictionary and save it to the save_path 


