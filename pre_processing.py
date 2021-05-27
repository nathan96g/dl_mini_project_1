import torch

def to_hot_encoding(x):
    """
    0 -> [1,0,0,0,0,0...]
    1 -> [0,1,0,0,0,0...]
    2 -> [0,0,1,0,0,0...]
    3 -> [0,0,0,1,0,0...]
    """
    identity = torch.eye(10)
    return identity[x]

def unzip_pairs(input, classes = None):
    """
    Extract left and right number (with associated classes)
    from the input dataset. 
    """
    left_input = input[:, 0, :, :]
    right_input = input[:, 1, :, :]
    if classes == None : #case when we want only to split the input
        return left_input, right_input
    left_classes = classes[:, 0]
    right_classes = classes[:, 1]
    return left_input, left_classes, right_input, right_classes

def merge_dataset(left_input, left_classes, right_input, right_classes):
    """
    Concat 2 datasets of the size dimensions into one dataset
    """
    input = torch.cat((left_input, right_input),0)
    classes = torch.cat((left_classes, right_classes),0)
    return input, classes

def unzip_and_merge(input, classes):
    """
    Returns a dataset of images instead of a dataset of 
    pair of images
    """
    left_input, left_classes, right_input, right_classes = unzip_pairs(input, classes)
    return merge_dataset(left_input, left_classes, right_input, right_classes)

def prepare_train_dataset(input, classes):
    """
    Returns a dataset of images instead of a dataset of 
    pair of images and classes labels are converted to 
    hot_encoding
    """
    new_input, new_classes = unzip_and_merge(input, classes)
    return new_input, to_hot_encoding(new_classes)