import os
from datetime import datetime

import torch.utils.data
from torch import optim
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from transformers import T5Tokenizer, T5EncoderModel
from torch.utils.data import DataLoader
import shutil
import time
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import os
import requests


from minimagen.Imagen import Imagen
from minimagen.Unet import Unet, Base, Super, BaseTest, SuperTest
from minimagen.generate import load_minimagen, load_params
from minimagen.t5 import get_encoded_dim
from minimagen.training import get_minimagen_parser, ConceptualCaptions, get_minimagen_dl_opts, \
    create_directory, get_model_params, get_model_size, save_training_info, get_default_args, MinimagenTrain, \
    load_restart_training_parameters, load_testing_parameters

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Command line argument parser. See `training.get_minimagen_parser()`.
parser = get_minimagen_parser()
# Add argument for when using `main.py`
parser.add_argument("-ts", "--TIMESTAMP", dest="timestamp", help="Timestamp for training directory", type=str,
                             default=None)
args = parser.parse_args()
timestamp = args.timestamp

# Get training timestamp for when running train.py as main rather than via main.py
if timestamp is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create training directory
dir_path = f"./training_{timestamp}"
training_dir = create_directory(dir_path)

# If loading from a parameters/training directory
if args.RESTART_DIRECTORY is not None:
    args = load_restart_training_parameters(args)
elif args.PARAMETERS is not None:
    args = load_restart_training_parameters(args, justparams=True)

# If testing, lower parameter values to lower computational load and also to lower amount of data being used.
def pull_img(img_filename):
    """Her kullanacağımızda bu kod üzerinden çağırıp resmi açacağız, batch batch yapacağımız için memoryi çok
    zorlamaması lazım bu şekilde, her batch bittiğinde bütün imagelar için img.close diyeceğiz"""
    filename = img_filename
    img = Image.open(filename)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


filename_test = "C:/Users/Mert Edgü/Desktop/neural/eee443_project_dataset_test.h5"
filename_train = "C:/Users/Mert Edgü/Desktop/neural/eee443_project_dataset_train.h5"

ftest = h5py.File(filename_test, 'r')
ftrain = h5py.File(filename_train, 'r')

train_cap = np.array(ftrain['train_cap'])
test_cap = np.array(ftest['test_caps'])
train_imid = np.array(ftrain['train_imid'])
test_imid = np.array(ftest['test_imid'])
train_url = np.array(ftrain['train_url'])
test_url = np.array(ftest['test_url'])
word_code = ftrain['word_code']
index = np.array(list(np.array(word_code)[0]), dtype=int)
names = np.array(word_code[0].dtype.names)

ftrain.close()
ftest.close()

args1 = np.argsort(index)
word_index = index[args1]
word_names = names[args1]

train_url_ids = np.arange(len(train_url)) + 1
url_id_pairs = zip(train_url, train_url_ids)

# download_parallel(url_id_pairs)

image_ids = []
invalid_ids = []
images_dict = {}
train_id_caption_idx = []
train_id_caption_names = []

for i in range(
        len(train_url)):  # her bir img_id için deneyecek klasörden çekmeyi, resim başta yoktuysa exceptiona girecek
    img_id = i + 1
    image_ids.append(img_id)

    filename = "C:/Users/Mert Edgü/Desktop/neural/train_data/" + str(img_id) + ".jpg"
    isExist = os.path.exists(filename)
    if isExist:
        # temp = Image.open(filename)
        # img = Image.open(filename)
        # if img.mode != "RGB":
        #    img = img.convert("RGB")
        images_dict[img_id] = filename
        indexes = np.where(train_imid == img_id)  # arguments where imid is equal to img_id
        captions = train_cap[indexes]
        for caption in captions:
            caption_index = caption
            caption_names = word_names[caption_index]
            element_name = (img_id, caption_names)
            element_idx = (img_id, caption_index)
            train_id_caption_names.append(element_name)
            train_id_caption_idx.append(element_idx)
            # element = (img, caption) #eğer img-caption tutarsak
            # train_img_caption.append(element)
        # img.close()
    else:
        invalid_ids.append(img_id)

# direkt kelimeleri kullanacaksak train_id_caption_idx i silebiliriz
# train_id_caption_idx captionları word_indexdeki indexler gibi tutuyor, train_id_caption_namesde captionlar
# kelime olarak tutuluyor, ikisinde de imagelar id olarak tutuluyor
# images_dictte artık filename var, pull_img kullanarak image ı klasörden çekip açacağız

# testde sadece idler farklı verilmiş olabilir, onlarda +-1 olayı olabilir, ya da farklı bir indexing yapılır vs
# image ve captionı çekmek için prototip bir kod, bu img_id ve caption tupleları tuttuğumuzda, direkt image ve imid de tutabiliriz listede


k = 0  # herhangi bir index, listin indexi, yani data instance gibi düşünebiliriz
im_id = train_id_caption_names[k][0]
im_caption_name = train_id_caption_names[k][1]
img_fn = images_dict[im_id]
img = pull_img(img_fn)

# buradan sonra image ı modele vereceğiz, forward pass yapıp sonra img.close()
# ya da bütün imageları img to tensor ile tensor olarak tutacağız, bunları bir pickle a kaydetmek gerekebilir böyle bir
# durumda, memory patlamasın diye yüzerli olarak tensor çevir, liste (ya da başka bir şeye) ekle yapabiliriz
# en sonda da pickle a ya da h5 e yazabiliriz, her seferinde de oradan çekeriz, resimlerle işimiz biter


MAX_LENGTH = 256

DEFAULT_T5_NAME = 't5_small'

# Variants: https://huggingface.co/docs/transformers/model_doc/t5v1.1. 1.1 versions must be finetuned.
T5_VERSIONS = {
    't5_small': {'tokenizer': None, 'model': None, 'handle': 't5-small', 'dim': 512, 'size': .24},
    't5_base': {'tokenizer': None, 'model': None, 'handle': 't5-base', 'dim': 768, 'size': .890},
    't5_large': {'tokenizer': None, 'model': None, 'handle': 't5-large', 'dim': 1024, 'size': 2.75},
    't5_3b': {'tokenizer': None, 'model': None, 'handle': 't5-3b', 'dim': 1024, 'size': 10.6},
    't5_11b': {'tokenizer': None, 'model': None, 'handle': 't5-11b', 'dim': 1024, 'size': 42.1},
    'small1.1': {'tokenizer': None, 'model': None, 'handle': 'google/t5-v1_1-small', 'dim': 512, 'size': .3},
    'base1.1': {'tokenizer': None, 'model': None, 'handle': 'google/t5-v1_1-base', 'dim': 768, 'size': .99},
    'large1.1': {'tokenizer': None, 'model': None, 'handle': 'google/t5-v1_1-large', 'dim': 1024, 'size': 3.13},
    'xl1.1': {'tokenizer': None, 'model': None, 'handle': 'google/t5-v1_1-xl', 'dim': 2048, 'size': 11.4},
    'xxl1.1': {'tokenizer': None, 'model': None, 'handle': 'google/t5-v1_1-xxl', 'dim': 4096, 'size': 44.5},
}


# Fast tokenizers: https://huggingface.co/docs/transformers/main_classes/tokenizer
def _check_downloads(name):
    if T5_VERSIONS[name]['tokenizer'] is None:
        T5_VERSIONS[name]['tokenizer'] = T5Tokenizer.from_pretrained(T5_VERSIONS[name]['handle'])
    if T5_VERSIONS[name]['model'] is None:
        T5_VERSIONS[name]['model'] = T5EncoderModel.from_pretrained(T5_VERSIONS[name]['handle'])


def t5_encode_text(text, name: str = 'google/t5-v1_1-base', max_length=MAX_LENGTH):
    _check_downloads(name)
    tokenizer = T5_VERSIONS[name]['tokenizer']
    model = T5_VERSIONS[name]['model']

    # Move to cuda is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
    else:
        device = torch.device('cpu')

    # Tokenize text
    tokenized = tokenizer.batch_encode_plus(
        text,
        padding='longest',
        max_length=max_length,
        truncation=True,
        return_tensors="pt",  # Returns torch.tensor instead of python integers
    )

    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    model.eval()

    # Don't need gradient - T5 frozen during Imagen training
    with torch.no_grad():
        t5_output = model(input_ids=input_ids, attention_mask=attention_mask)
        final_encoding = t5_output.last_hidden_state.detach()

    # Wherever the encoding is masked, make equal to zero
    final_encoding = final_encoding.masked_fill(~rearrange(attention_mask, '... -> ... 1').bool(), 0.)

    return final_encoding, attention_mask.bool()


def get_encoded_dim(name: str) -> int:
    """
    Gets the encoding dimensionality of a given T5 encoder.
    """
    return T5_VERSIONS[name]['dim']



class NeuralDataset(torch.utils.data.Dataset):
    def __init__(self, id_caption, image_dict, encoder_name: str, max_length: int,
                 side_length: int,  img_transform=None):

        self.ids = []
        self.captions = []
        self.image_dict = image_dict

        for i in range(len(id_caption)):
            self.ids.append(id_caption[i][0])
            self.captions.append(list(id_caption[i][1]))

        if img_transform is None:
            self.img_transform = Compose([Resize((side_length, side_length)), ToTensor()])
        else:
            self.img_transform = Compose([Resize((side_length, side_length)), ToTensor(), img_transform])
        self.encoder_name = encoder_name
        self.max_length = max_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = pull_img(self.image_dict[self.ids[idx]])
        if img is None:
            return None
        elif self.img_transform:
            img = self.img_transform(img)

        # Have to check None again because `Resize` transform can return None
        if img is None:
            return None
        elif img.shape[0] != 3:
            return None

        enc, msk = t5_encode_text(self.captions[idx], self.encoder_name, self.max_length)

        return {'image': img, 'encoding': enc, 'mask': msk}


def TrainTestSplit(dset, images_dict):

    # Torch train/valid dataset
    dataset_train_valid = NeuralDataset(dset, images_dict, 't5_base', 128, 64)

    # Split into train/valid
    train_size = int(0.8 * len(dataset_train_valid))
    valid_size = len(dataset_train_valid) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset_train_valid, [train_size, valid_size])

    return train_dataset, valid_dataset


train_dataset, valid_dataset = TrainTestSplit(train_id_caption_names, images_dict)


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=True)

# Create Unets
if args.RESTART_DIRECTORY is None:
    imagen_params = dict(
        image_sizes=(int(args.IMG_SIDE_LEN / 2), args.IMG_SIDE_LEN),
        timesteps=args.TIMESTEPS,
        cond_drop_prob=0.15,
        text_encoder_name=args.T5_NAME
    )

    # If not loading a training from a checkpoint
    if args.TESTING:
        # If testing, use tiny MinImagen for low computational load
        unets_params = [get_default_args(BaseTest), get_default_args(SuperTest)]

    # Else if not loading Unet/Imagen settings from a config (parameters) folder, use defaults
    elif not args.PARAMETERS:
        # If no parameters provided, use params from minimagen.Imagen.Base and minimagen.Imagen.Super built-in classes
        unets_params = [get_default_args(Base), get_default_args(Super)]

    # Else load unet/Imagen configs from config (parameters) folder (override imagen+params)
    else:
        # If parameters are provided, load them
        unets_params, imagen_params = get_model_params(args.PARAMETERS)

    # Create Unets accoridng to unets_params
    unets = [Unet(**unet_params).to(device) for unet_params in unets_params]

    # Create Imagen from UNets with specified imagen parameters
    imagen = Imagen(unets=unets, **imagen_params).to(device)
else:
    # If training is being resumed from a previous one, load all relevant models/info (load config AND state dicts)
    orig_train_dir = os.path.join(os.getcwd(), args.RESTART_DIRECTORY)
    unets_params, imagen_params = load_params(orig_train_dir)
    imagen = load_minimagen(orig_train_dir).to(device)
    unets = imagen.unets

# Fill in unspecified arguments with defaults for complete config (parameters) file
unets_params = [{**get_default_args(Unet), **i} for i in unets_params]
imagen_params = {**get_default_args(Imagen), **imagen_params}

# Get the size of the Imagen model in megabytes
model_size_MB = get_model_size(imagen)

# Save all training info (config files, model size, etc.)
save_training_info(args, timestamp, unets_params, imagen_params, model_size_MB, training_dir)

# Create optimizer
optimizer = optim.Adam(imagen.parameters(), lr=args.OPTIM_LR)

# Train the MinImagen instance
MinimagenTrain(timestamp, args, unets, imagen, train_dataloader, valid_dataloader, training_dir, optimizer, timeout=30)