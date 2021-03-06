from torch import nn
from torch.nn.modules.loss import MSELoss
import data_augmentation as data
import models as m
import advanced_models as am
import torch
import dlc_practical_prologue as prologue
import processing as proc

torch.manual_seed(42)

# Extract train and test dataset
train_input, train_target, train_classes, test_input, test_target, test_classes = \
    prologue.generate_pair_sets(1000)

print("Befor pre processing:")
print("Input size: ", train_input.size())
print("targets size ", train_classes.size())

img_size = (14, 14)
img_vect_length = img_size[0] * img_size[1]
data_size = train_input.size()[0] * train_input.size()[1] #1000 * 2 images
max_pixel_val = 255 #pixels can take value from 0 to 255

# Can be pass for the training, will augment the dataset size with
# twisted images (shift, noise and block).
#augmenter = data.Augmenter(img_size, max_pixel_val, data_size)
augmenter = None

hidden_units = 100
nb_epochs = 51
learning_rate = 1e-1 
mini_batch_size = 200

models = [
            m.Net_0(150, img_size, batch_normalization=False),
            m.Net_1(100, batch_normalization=False),
            m.Net_2(100, batch_normalization=False),
            #am.LeNet(130),
            #am.ResNet(nb_residual_blocks = 5, 
            #          nb_channels = 10, 
            #          kernel_size = 3, 
            #          nb_classes = 10, 
            #          img_size=(14,14)),
            #am.ResNeXt(filters=42, nb_blocks=3, width=2, cardinality=2, img_size=(14,14))
        ]

criterions = [nn.MSELoss(), nn.CrossEntropyLoss()]

results = [['Model', 'Batch Normalization', 'Loss', 'Accuracy: Max Pair', 'Accuracy: Joint Probability']]

for model in models:
  for criterion in criterions:

    print("*****************************************************")
    print(model.__class__.__name__ +" with "+ criterion.__class__.__name__)
    #print(model)
    optimizer = torch.optim.Adam(model.parameters())

    model = proc.train_model(model, 
                    train_input, train_target, train_classes, 
                    test_input, test_target, test_classes, 
                    criterion, optimizer, 
                    nb_epochs, mini_batch_size, augmenter, verbal=True)
   
    acc_max = proc.accuracy_comparisons(model, test_input, test_target, mini_batch_size, rule="no_proba")
    acc_joint = proc.accuracy_comparisons(model, test_input, test_target, mini_batch_size, rule="joint_proba")
    
    results.append([model.__class__.__name__, 
                    model.batch_normalization, 
                    criterion.__class__.__name__, 
                    acc_max,
                    acc_joint])
    
    print("accuracy comparison with max pair : {}".format(acc_max))
    print("accuracy comparison with joint probability : {}".format(acc_joint))
