from .pipeline import Pipeline, register

def convertMask(masks):
  """this function get the mask tensor (batchsize, 1, H, W) and returns a target tensor 
  with dimension (batchsize, 2, H, W), which is a ground truth probability.
  2 is the class number
  """
  return torch.cat(((masks == 0).float(),(masks == 255).float()), 0)

class ImageDataset(Dataset):
    def __init__(self, input_dir, transform = None):
      self.input_dir = input_dir
      self.transform = transform
      self.trans = transforms.ToTensor()
      self.image = []
      self.mask = []
      self.num = [] # the number of active channel
    def transformation(self, images, masks):
      if torch.randperm(2)[0]:
        images = TF.hflip(images)
        masks = TF.hflip(masks)
      if torch.randperm(2)[0]:
        images = TF.vflip(images)
        masks = TF.vflip(masks)
      
      return self.trans(images), self.trans(masks)*255

    def __len__ (self):
        return sum([str.isdigit(s) for s in os.listdir(self.input_dir)])

    def __getitem__(self, idx):
      path_image = self.input_dir + str(idx) + "/"+'image.jpg'
      path_mask = self.input_dir + str(idx) + "/"+'mask.png'
      
      if os.path.exists(path_image) and os.path.exists(path_mask):
        self.image = Image.open(path_image)
        self.mask = Image.open(path_mask)

      if self.transform:
        self.image, self.mask = self.transformation(self.image, self.mask)
      else:
        self.image = self.trans(self.image)
        self.mask = (self.trans(self.mask)).long()
      
      #print(self.mask.shape)
      self.mask = self.mask[0,:,:].reshape(1,256,384)

      return self.image, self.mask, 


# Functions for adding the convolution layer
def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
  if useBN:
    # Use batch normalization
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out,),
      nn.LeakyReLU(0.1)
    )
  else:
    # No batch normalization
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU(),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU()
    )

# Upsampling
def upsample(ch_coarse, ch_fine):
  return nn.Sequential(
    nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
    nn.ReLU()
  )


# U-Net
class unet(nn.Module):
  def __init__(self, useBN=False):
    super(unet, self).__init__()
    # Downgrade stages
    self.conv1 = add_conv_stage(3, 32, useBN=useBN)
    self.conv2 = add_conv_stage(32, 64, useBN=useBN)
    self.conv3 = add_conv_stage(64, 128, useBN=useBN)
    self.conv4 = add_conv_stage(128, 256, useBN=useBN)
    # Upgrade stages
    self.conv3m = add_conv_stage(256, 128, useBN=useBN)
    self.conv2m = add_conv_stage(128,  64, useBN=useBN)
    self.conv1m = add_conv_stage( 64,  32, useBN=useBN)
    # Maxpool
    self.max_pool = nn.MaxPool2d(2)
    # Upsample layers
    self.upsample43 = upsample(256, 128)
    self.upsample32 = upsample(128,  64)
    self.upsample21 = upsample(64 ,  32)
    #TODO: Design your last layer & activations
    self.conv4m= add_conv_stage(32, 32, useBN=useBN)
    self.conv_Final = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)

    # weight initialization
    # You can have your own weight intialization. This is just an example.
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if m.bias is not None:
          m.bias.data.zero_()

  def forward(self, x):
    conv1_out = self.conv1(x)
    conv2_out = self.conv2(self.max_pool(conv1_out))
    conv3_out = self.conv3(self.max_pool(conv2_out))
    conv4_out = self.conv4(self.max_pool(conv3_out))

    conv4m_out_ = torch.cat((self.upsample43(conv4_out), conv3_out), 1)
    conv3m_out  = self.conv3m(conv4m_out_)

    conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
    conv2m_out  = self.conv2m(conv3m_out_)

    conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
    conv1m_out  = self.conv1m(conv2m_out_)
     
    conv1m_out = self.conv4m(conv1m_out)
    onv1m_out = self.conv4m(conv1m_out)

    output = self.conv_Final(conv1m_out)
 
    return output

def training_step(model):
  model.train()
  #form the random index
  index = torch.randperm(len(train_dataset))[:train_batch_size].data.numpy()
  
  #get the first index
  image_batch = train_dataset[index[0]][0].reshape(1, 3, 256, 384)
  label_batch = train_dataset[index[0]][1].reshape(1, 1, 256, 384)

  for i in range(len(index)-1):
    a = train_dataset[index[i+1]][0].reshape(1, 3, 256, 384)
    b = train_dataset[index[i+1]][1].reshape(1, 1, 256, 384)
    image_batch = torch.cat((image_batch, a), dim = 0)
    label_batch = torch.cat((label_batch, b), dim = 0)

  if gpu_bool:
    image_batch = image_batch.cuda()
    label_batch = label_batch.cuda()
    #print(image_batch.shape, label_batch.shape)
   
  optimizer.zero_grad()
  output = model(image_batch)
  loss = train_loss(output, label_batch)

  loss.backward()
  optimizer.step()
  #scheduler.step()

  pred = MakePrediction(output)
  #print(pred.shape)

  return loss, DiceScore(pred, label_batch), Accuracy(pred, label_batch)

def val_step(model):
  model.eval()
  #form the random index
  index = torch.randperm(len(val_dataset))[:validation_batch_size].data.numpy()
  #get the first index
  image_batch = val_dataset[index[0]][0].reshape(1, 3, 256, 384)
  label_batch = val_dataset[index[0]][1].reshape(1, 1, 256, 384)
  for i in range(len(index)-1):
    a = val_dataset[index[i+1]][0].reshape(1, 3, 256, 384)
    b = val_dataset[index[i+1]][1].reshape(1, 1, 256, 384)
    image_batch = torch.cat((image_batch, a), dim = 0)
    label_batch = torch.cat((label_batch, b), dim = 0)

  if gpu_bool:
    image_batch = image_batch.cuda()
    label_batch = label_batch.cuda()
    
  output = model(image_batch)
  vali_loss = val_loss(output, label_batch)

  pred = MakePrediction(output)

  return vali_loss, DiceScore(pred, label_batch), Accuracy(pred, label_batch)

def MakePrediction(output):
  """this function get the model output tensor (batchsize, 8, H, W) (without softmax) and returns a prediction tensor
  with dimension (batchsize, 2, H, W), with values of 0 or 1.
  2 is the class number
  """
  softmax = nn.Softmax(dim=1)
  _,d = torch.max(softmax(output), 1, keepdim = True)
  return torch.cat(((d == 0).float(),(d == 1).float()), 1)

def Accuracy(pred, target):
  #target is ground truth label
  #pred is prediction (the output from the MakePrediction)
  batch_size, num_class, H, W = pred.shape
  Accuracy_batch = 0
  for i in range(batch_size):#for each example
    #just use the vehicle channel to predict the accuracy (not background)
    a = pred[i, 1, :, :].float()
    b = target[i, 0, :, :].float()
    Accuracy_batch += torch.sum((torch.abs(a - b) < 1E-5).float())/H/W
  
  Accuracy_batch /= batch_size
  return Accuracy_batch

def DiceScore(pred, target):
  #target is ground truth label
  #pred is prediction (the output from the MakePrediction)
  batch_size, num_class, H, W = pred.shape
  #print(pred.size())
  alpha = 0.0001
  Dice_batch = 0
  for i in range(batch_size):#for each example
    Dice_channel = 0
    #for the vehicle channel
    a = torch.sum(pred[i,1] * target[i,0])
    b = torch.sum(pred[i,1] + target[i,0])
    Dice_channel = ((2*a)/(b))
    Dice_batch += Dice_channel

  return Dice_batch/batch_size

class DICELoss(nn.Module):
  def __init__(self, batch_size = 10, class_num = 2, epilson = 0.0001):
      super(DICELoss, self).__init__()
      self.BS = batch_size
      self.Classnum = class_num
      self.epilson = epilson

  def forward(self, output, target):
    loss_batch = 0
    #pass the output from the unet through a softmax
    softmax = nn.Softmax(dim=1)
    output = softmax(output)
    #print(output.max().item(),output.min().item())
    #print(target.max().item(),target.min().item())
    for i in range(self.BS): #for each example
        #for the vehicle channel
        a = torch.sum(output[i,1,:,:] * target[i,0,:,:])
        b = torch.sum(output[i,1,:,:] + target[i,0,:,:])
        loss_batch += (1 - (2*a)/(b))

    return loss_batch/self.BS


@register
class Segmentation(Pipeline):
    """This is the segmentation pipeline for stage 1"""
    def load_data(data_path=None, train_batch_size=10):
        if not data_path:
            get_ipython().system_raw('/usr/sbin/sshd -D &')
            data_path = "./Resized_Img_Mask"
        
        self.train_dataset = ImageDataset(train_path, transform=False)
