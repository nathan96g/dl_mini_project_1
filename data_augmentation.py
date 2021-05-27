import torch
import pre_processing

class Augmenter:

    def __init__(self, img_size, max_pixel_val, data_size):
        self.img_size = img_size
        self.max_pixel_val = max_pixel_val
        self.data_size = data_size

    def noise(self, input, intensity):
        res = input + torch.empty(self.img_size).normal_(0,intensity)
        return res.clamp_(0, self.max_pixel_val).to(int)


    def noise_mask(self, perc_pixel_changed):
        mask = torch.empty(self.img_size).uniform_(0 -perc_pixel_changed, 1-perc_pixel_changed)
        mask = torch.where(mask < 0, torch.ones(self.img_size), torch.zeros(self.img_size))
        return mask

    def noise_custom(self, input, arg):
      intensity = arg[0]
      perc_pixel_changed = arg[1]
      res = input + torch.empty(self.img_size).normal_(0,intensity) * self.noise_mask(perc_pixel_changed)
      return res.clamp_(0, self.max_pixel_val).to(int)


    ###### Function block ######
    def block(self, input, block_size):
      coordinate_of_block = torch.randint(0, 6, (2, 1)) + 3
      x_block = coordinate_of_block[1]
      y_block = coordinate_of_block[0]

      square_block = torch.ones(block_size, block_size) * -1
      block_matrix = torch.ones(self.img_size) 
      block_matrix[y_block:y_block+block_size, x_block:x_block+block_size] += square_block

      return input * block_matrix


    ###### Function shift ###### 
    def shift(self, input,shift_size):
      #random coordinate of the x and y shift 
      shift_coordinate = torch.randint(-shift_size, shift_size+1, (2, 1))
      x_shift = shift_coordinate[1]
      y_shift = shift_coordinate[0]

      large_tensor = torch.zeros(input.shape[0],22,22)
      large_tensor[:,4:4+self.img_size[0], 4:4+self.img_size[1]] += input

      return large_tensor[:,4+y_shift : 18+y_shift, 4+x_shift : 18+x_shift]

    def compute_by_batch(self, input_batch, func, arg):
      output_shift = torch.empty(0)
      batch_size = int(input_batch.size(0)/10)

      for b in range(0, input_batch.size(0), batch_size):
        output_minibatch_shift = func(input_batch[b:b+batch_size], arg)
        output_shift= torch.cat((output_shift, output_minibatch_shift), 0)
      return output_shift

    def data_augmentation_(self, split_input_data, split_input_classes, percentage_pert, func, arg):

        if percentage_pert < 0.005 or percentage_pert > 1 : percentage = 0

        nbr_pert = round(self.data_size * percentage_pert * 0.1) * 10
        index_pert = torch.randperm(self.data_size - 1)[0 : nbr_pert]

        if percentage_pert != 0 : 
          split_input_data[index_pert]= self.compute_by_batch(split_input_data[index_pert], func, arg)

        return


    def data_augmentation_full(self, input_data, input_classes,  
                          percentage_shift=0,
                          percentage_noise=0,
                          percentage_block=0
                          ): 

        split_input_data,split_input_classes = pre_processing.unzip_and_merge(input_data,input_classes)

        shift_factor = 3
        self.data_augmentation_(split_input_data, split_input_classes, percentage_shift, self.shift, shift_factor)

        intensity_noise = 50
        self.data_augmentation_(split_input_data, split_input_classes, percentage_noise, self.noise, intensity_noise)

        block_size = torch.randint(3, 5, (1,))
        self.data_augmentation_(split_input_data, split_input_classes, percentage_block, self.block, block_size)

        input_data = torch.reshape(split_input_data, [1000, 2, 14, 14])  
        input_classes = torch.reshape(split_input_classes, [1000, 2])

        return input_data,input_classes


    def data_augmentation(self, split_input_data, split_input_classes, percentage_pert, func, arg):

        percentage = 0 if percentage_pert < 0.005 or percentage_pert > 1 else percentage_pert

        nbr_pert = round(self.data_size * percentage * 0.1) * 10
        index_pert = torch.randperm(self.data_size - 1)[0 : nbr_pert]

        if percentage != 0 :
            return self.compute_by_batch(split_input_data[index_pert], func, arg), \
                   split_input_classes[index_pert]

        return split_input_data[index_pert], \
               split_input_classes[index_pert]


    def __call__(self, input_data, input_classes,  
                          percentage_shift=0.2,
                          percentage_noise=0.2,
                          percentage_block=0.2
                          ):
        """
        This function return an augmented version of the input_data
        given as argument. It returns:
            input_data + 
            shift(percentage_shift % of input_data) + 
            noise(percentage_noise % of input_data) +
            block(percentage_block % of input_data)
        """
        shift_factor = 3
        shifted, shifted_classes = self.data_augmentation(input_data, 
                                                          input_classes, 
                                                          percentage_shift, 
                                                          self.shift, 
                                                          shift_factor)

        intensity_noise = 50
        noise, noise_classes = self.data_augmentation(input_data, 
                                                      input_classes, 
                                                      percentage_noise, 
                                                      self.noise, 
                                                      intensity_noise)

        block_size = torch.randint(3, 5, (1,))
        block, block_classes = self.data_augmentation(input_data, 
                                                      input_classes, 
                                                      percentage_block, 
                                                      self.block, 
                                                      block_size)

        tmp, tmp_classes = pre_processing.merge_dataset(shifted, shifted_classes, 
                                                        noise, noise_classes)
        tmp, tmp_classes = pre_processing.merge_dataset(tmp, tmp_classes, 
                                                        block, block_classes)

        return pre_processing.merge_dataset(tmp, tmp_classes, input_data, input_classes)