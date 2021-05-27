import torch
import pre_processing
from torch import nn
from torch.nn import functional as F


def accuracy_numbers(model, input, classes, mini_batch_size):
    """
    accuracy w.r.t. the image samples (if each sample image is correctly classified or not)
    !!! input must be already just the one sample form and not pairs of sample form
    !!! classes must be already in hot encoding form
    """
    pred_numbers = predict_numbers(model, input, mini_batch_size, probability_classes = False)
    result = 1 - (classes - pred_numbers).abs().sum() / (classes.shape[0] * 2)  #since if there is an error, it will be counted twice
    return round(result.item(), 7) 


def numbers_accuracy_to_string(model, train_input, train_classes, test_input, test_classes, mini_batch_size):
    """
    Compute accuracy for train and test dataset and pretty print it
    """
    train_accuracy = accuracy_numbers(model, train_input, train_classes, mini_batch_size)
    test_accuracy = accuracy_numbers(model, test_input, test_classes, mini_batch_size)
    return "train : {} ,  test : {}".format(train_accuracy, test_accuracy), train_accuracy,test_accuracy


def predict_numbers(model, input, mini_batch_size, probability_classes=False):
    """
    The robability_classes boolean idicate we return the probability
    of sample to belong to one class or the actual prediction
    """
    predictions = torch.zeros(input.size(0), 10)
  
    for b in range(0, input.size(0), mini_batch_size):
      output = model(input.narrow(0, b, mini_batch_size))
      if probability_classes:
        predictions[b: b + mini_batch_size] = F.softmax(output, dim=1)
      else : 
        _, res_indices = torch.max(output, 1)
        predictions[b: b + mini_batch_size] = pre_processing.to_hot_encoding(res_indices)
    return predictions


def predict_comparisons(model, left_input, right_input, mini_batch_size, rule="no_proba"):
    """
    Rule : decide the way to compare :
        "no_proba":    foreach pair, only the most likely class foreach image is kept
        "joint_proba": foreach pair, we keep the probability foreach image to belong to one particular class,
                       then we do foreach pair the joint probability to get the most likely right comparison, i.e.
                       we compute Pr(Left < Right)

    Note: we assume that the left and the right input are independent 
    (which it is not necessary the case since they have been trained on same network) 
    """
    probability_classes = rule != "no_proba"

    pred_left = predict_numbers(model, left_input, mini_batch_size, probability_classes)
    pred_right = predict_numbers(model, right_input, mini_batch_size, probability_classes)

    if rule == "no_proba" : 
      return (pred_left.argmax(dim=1) <= pred_right.argmax(dim=1)).int()
    elif rule == "joint_proba":
      result = torch.zeros((pred_left.shape[0]))
      for r_val in range(10): 
        result += pred_left[:, 0 : r_val + 1].sum(dim = 1) * pred_right[:, r_val]
      return (result >= 0.5).int()
    else : raise ValueError("Unknown rule, only 'no_proba' or 'joint_proba' are valid")


def accuracy_comparisons(model, input, target, mini_batch_size, rule="no_proba"):
    """
    Accuracy w.r.t. the comparison output
    """
    left_input, right_input = pre_processing.unzip_pairs(input)
    left_input = left_input.unsqueeze(1)
    right_input = right_input.unsqueeze(1)

    pred_comparisons = predict_comparisons(model, left_input, right_input, mini_batch_size, rule=rule)
    return 1 - (target - pred_comparisons).abs().sum() / target.shape[0]

def train_model_standard(model, 
                         train_input, train_classes,
                         test_input, test_classes,
                         criterion, optimizer, nb_epochs, mini_batch_size,
                         CrossEntropy=True, verbal=True):


    hot_train_classes = pre_processing.to_hot_encoding(train_classes)
    hot_test_classes = pre_processing.to_hot_encoding(test_classes)

    train_input = train_input.unsqueeze(1)
    test_input = test_input.unsqueeze(1)

    if verbal : print("training model ...")

    for e in range(nb_epochs):
        acc_loss = 0
        # with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(torch.narrow(train_input,0,b,mini_batch_size))

            if CrossEntropy : loss = criterion(output, train_classes.narrow(0, b, mini_batch_size))
            else : loss = criterion(output, hot_train_classes.narrow(0, b, mini_batch_size))

            acc_loss = acc_loss + loss.item()
            optimizer.zero_grad() #model set through the optimizer
            loss.backward() #accumulate gradient
            optimizer.step()

        if verbal and e % 2 == 0 :
            result = "epoch : {},  loss : {}  ".format(e, round(acc_loss, 8))
            add_result, train_accuracy,test_accuracy= numbers_accuracy_to_string(model, train_input, hot_train_classes, test_input, hot_test_classes, mini_batch_size)
            train_accuracy
            result += add_result
            print(result)
    
    if verbal : print("training model terminated.")  
    
    return model

def train_model(model, 
                train_input, train_target, train_classes, 
                test_input, test_target, test_classes, 
                criterion, optimizer, nb_epochs, 
                mini_batch_size, verbal=True):
    """
    Train the given model with input and classes given as argument
    """  
    merge_train_input, merge_train_classes =  pre_processing.unzip_and_merge(train_input, train_classes)
    merge_test_input, merge_test_classes =  pre_processing.unzip_and_merge(test_input, test_classes)

    CrossEntropy = type(criterion) is nn.CrossEntropyLoss
    
    return train_model_standard(model,
                                merge_train_input, merge_train_classes,
                                merge_test_input, merge_test_classes, 
                                criterion, optimizer, 
                                nb_epochs, mini_batch_size,CrossEntropy, verbal)