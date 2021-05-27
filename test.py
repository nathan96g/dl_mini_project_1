from data_augmentation import DataAugmentation
from torch import nn
import models as m
import advanced_models as am
import torch
import dlc_practical_prologue as prologue
import processing as proc

torch.manual_seed(42)

# Extract train and test dataset
train_input, train_target, train_classes, test_input, test_target, test_classes = \
    prologue.generate_pair_sets(1000)

print(train_input.size())
print(train_classes.size())

img_size = (14, 14)
img_vect_length = img_size[0] * img_size[1]
data_size = train_input.size()[0] * train_input.size()[1] #1000 * 2 images
max_pixel_val = 255 #pixels can take value from 0 to 255

# Augment train dataset
augmenter = DataAugmentation(img_size, max_pixel_val, data_size)

hidden_units = 100
nb_epochs = 51
learning_rate = 1e-1 
mini_batch_size = 200

models = [
          # Net_0(100, batch_normalization=False),
          # Net_0(120, batch_normalization=True),
          # Net_0(140, batch_normalization=True),
          #m.Net_0(150, img_size, batch_normalization=True),
          
          #m.Net_0(150, img_size, batch_normalization=False),

          # Net_1(70, batch_normalization=True),
          # Net_1(80, batch_normalization=True),
          
          # m.Net_1(100, batch_normalization=False),
          
          # Net_1(120, batch_normalization=True),
          # Net_1(140, batch_normalization=True),

          # Net_2(80, batch_normalization=True),
          m.Net_2(100, batch_normalization=False),
          m.Net_2(100, batch_normalization=True),
          # Net_2(110, batch_normalization=True),
          # Net_2(120, batch_normalization=True),
          # Net_2(130, batch_normalization=True),
          # LeNet(3),
          # LeNet(10),
          # LeNet(15),
          # LeNet(70),
          # LeNet(80),
          # LeNet(90),
          # LeNet(100),
          # LeNet(110),
          # am.LeNet(120),
          am.LeNet(130),
          # LeNet(120),
          # LeNet(130),
          # LeNet(140),
          # LeNet(150),
          #am.ResNet(nb_residual_blocks = 5, nb_channels = 10, kernel_size = 3, nb_classes = 10),
          #am.ResNeXt(filters=42, nb_blocks=3, width=2, cardinality=5)
          ]
          
criterions = [nn.CrossEntropyLoss()]

results = [['Model', 'Batch Normalization', 'Loss', 'Accuracy: Max Pair', 'Accuracy: Joint Probability']]


#if data augmentation :
#train_input,train_classes = data_augmentation_full(train_input, train_classes, percentage_shift=0, percentage_noise=0.5,percentage_block=0) 


for model in models:
  for criterion in criterions:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    # model = Net_1(100).to(device)

    print("*****************************************************")
    print(model.__class__.__name__ +" with "+ criterion.__class__.__name__)
    print(model)
    optimizer = torch.optim.Adam(model.parameters())

    model = proc.train_model(model, 
                    train_input, train_target, train_classes, 
                    test_input, test_target, test_classes, 
                    criterion, optimizer, 
                    nb_epochs, mini_batch_size, verbal=False)
  
    # data_julien.append(data_julien_sol) 
   
    acc_max = proc.accuracy_comparisons(model, test_input, test_target, mini_batch_size, rule="no_proba")
    acc_joint = proc.accuracy_comparisons(model, test_input, test_target, mini_batch_size, rule="joint_proba")
    
    results.append([model.__class__.__name__, 
                    model.batch_normalization, 
                    criterion.__class__.__name__, 
                    acc_max,
                    acc_joint])
    
    print("accuracy comparison with max pair : {}".format(acc_max))
    print("accuracy comparison with joint probability : {}".format(acc_joint))
