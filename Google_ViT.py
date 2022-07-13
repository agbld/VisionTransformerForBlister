#%%
# import libraries
import argparse
import os
import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from utils.dataset import TripletBlister_Dataset, Prototype_Dataset
from utils.losses import TripletLoss, TripletLoss_fix
from utils.networks import TripletNet
from tqdm import tqdm

#%%
# settings and hyperparameters
if __name__ == '__main__':
    # settings
    CUDA_AVAILABLE = torch.cuda.is_available()
    print('CUDA available:', CUDA_AVAILABLE)
    TRAIN_PATH    = './data/train_50'
    TEST_PATH     = './data/test_50'
    
    LOAD_MODEL    = True
    MODEL_PATH    = './models/tmp.pt'
    SAVE_EPOCHS = 1                     # number of epochs to save model
    NUM_WORKERS = 4                     # number of workers for data loader

    # hparam
    IMG_SIZE = 416                      # image size (side length), default: 512
    PATCH_SIZE = int(IMG_SIZE / 13)     # patch size (num of patch = patch_size ** 2), default: 4
    DIM = 1024                          # total dimension of q, k, v vectors in each head (dim. of single q, k, v = dim // heads), default: 64
    DEPTH = 8                           # number of attention layers, default: 6
    HEADS = 8                           # number of attention heads, default: 8
    OUTPUT_DIM = 1024                   # dimension of output embedding, default: 128
    LEARNING_RATE = 0.0001              # learning rate
    BATCH_SIZE_TRAIN = 64               # batch size for training
    BATCH_SIZE_TEST = 64                # batch size for testing
    N_EPOCHS = 150                      # number of epochs
    
    try:
        # argument parser (for command line arguments)
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_path', type=str, default=TRAIN_PATH)
        parser.add_argument('--test_path', type=str, default=TEST_PATH)
        parser.add_argument('--img_size', type=int, default=IMG_SIZE)
        parser.add_argument('--patch_size', type=int, default=PATCH_SIZE)
        parser.add_argument('--dim', type=int, default=DIM)
        parser.add_argument('--depth', type=int, default=DEPTH)
        parser.add_argument('--heads', type=int, default=HEADS)
        parser.add_argument('--output_dim', type=int, default=OUTPUT_DIM)
        parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
        parser.add_argument('--batch_size_train', type=int, default=BATCH_SIZE_TRAIN)
        parser.add_argument('--batch_size_test', type=int, default=BATCH_SIZE_TEST)
        parser.add_argument('--n_epochs', type=int, default=N_EPOCHS)
        args = parser.parse_args()
        
        # assign arguments to variables
        TRAIN_PATH    = args.train_path
        TEST_PATH     = args.test_path
        IMG_SIZE      = args.img_size
        PATCH_SIZE    = args.patch_size
        DIM           = args.dim
        DEPTH         = args.depth
        HEADS         = args.heads
        OUTPUT_DIM    = args.output_dim
        LEARNING_RATE = args.learning_rate
        BATCH_SIZE_TRAIN = args.batch_size_train
        BATCH_SIZE_TEST = args.batch_size_test
        N_EPOCHS      = args.n_epochs
        
    except:
        # use default settings if no command line arguments
        print('arguments error, use default settings')
    
    print('\nArguments:')
    print('TRAIN_PATH:', TRAIN_PATH)
    print('TEST_PATH:', TEST_PATH)
    print('IMG_SIZE:', IMG_SIZE)
    print('PATCH_SIZE:', PATCH_SIZE)
    print('DIM:', DIM)
    print('DEPTH:', DEPTH)
    print('HEADS:', HEADS)
    print('OUTPUT_DIM:', OUTPUT_DIM)
    print('LEARNING_RATE:', LEARNING_RATE)
    print('BATCH_SIZE_TRAIN:', BATCH_SIZE_TRAIN)
    print('BATCH_SIZE_TEST:', BATCH_SIZE_TEST)
    print('N_EPOCHS:', N_EPOCHS)

#%%
# model definition
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)
        self.do2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)
        
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = True) # Wq,Wk,Wv for each vector, thats why *3
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)
        
        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)        
        self.do1 = nn.Dropout(dropout)
        

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x) #gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1) #follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v) #product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)') #concat heads into one matrix, ready for next encoder block
        out =  self.nn1(out)
        out = self.do1(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attention, mlp in self.layers:
            x = attention(x, mask = mask) # go to attention
            x = mlp(x) #go to MLP_Block
        return x

# modified for triplet loss
class ImageTransformer(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dropout = 0.1, emb_dropout = 0.1):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2  # e.g. (32/4)**2= 64
        patch_dim = channels * patch_size ** 2  # e.g. 3*8**2 = 64*3

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_patches + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std = .02) # initialized based on the paper
        self.patch_conv= nn.Conv2d(3,dim, patch_size, stride = patch_size) #eqivalent to x matmul E, E= embedd matrix, this is the linear patch projection
        
        #self.E = nn.Parameter(nn.init.normal_(torch.empty(BATCH_SIZE_TRAIN,patch_dim,dim)),requires_grad = True)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) #initialized based on the paper
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        # self.nn1 = nn.Linear(dim, num_classes)  # if finetuning, just use a linear layer without further hidden layers (paper)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        
        # self.af1 = nn.GELU() # use additinal hidden layers only when training on large datasets
        # self.do1 = nn.Dropout(dropout)
        # self.nn2 = nn.Linear(mlp_dim, num_classes)
        # torch.nn.init.xavier_uniform_(self.nn2.weight)
        # torch.nn.init.normal_(self.nn2.bias)
        # self.do2 = nn.Dropout(dropout)

    def forward(self, img, mask = None):
        p = self.patch_size

        x = self.patch_conv(img) # each of 64 vecotrs is linearly transformed with a FFN equiv to E matmul
        #x = torch.matmul(x, self.E)
        x = rearrange(x, 'b c h w -> b (h w) c') # 64 vectors in rows representing 64 patches, each 64*3 long

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x, mask) #main game

        x = self.to_cls_token(x[:, 0])
        
        # remove class layer
        # x = self.nn1(x)
        
        # x = self.af1(x)
        # x = self.do1(x)
        # x = self.nn2(x)
        # x = self.do2(x)
        
        return x

# %%
# initialize dataloader
if __name__ == '__main__':
    # data transform for blister images
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([IMG_SIZE, IMG_SIZE]),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        torchvision.transforms.RandomAffine(8, translate=(.15, .15)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]), ])

    # training dataset/dataloader
    ps1_ps2_train_dataset = torchvision.datasets.ImageFolder(
        root=TRAIN_PATH, transform=data_transform)
    # ps1_ps2_train_loader = DataLoader(
    #     ps1_ps2_train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=NUM_WORKERS)

    # testing dataset/dataloader
    ps1_ps2_test_dataset = torchvision.datasets.ImageFolder(
        root=TEST_PATH, transform=data_transform)
    # ps1_ps2_test_loader = DataLoader(
    #     ps1_ps2_test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=NUM_WORKERS)

    # Triplet train dataset/dataloader
    triplet_train_dataset = TripletBlister_Dataset(
        ps1_ps2_train_dataset)  # Returns triplets of images
    kwargs = {'num_workers': NUM_WORKERS, 'pin_memory': True} if CUDA_AVAILABLE else {}
    triplet_train_loader = DataLoader(
        triplet_train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, **kwargs)

    # Triplet test dataset/dataloader
    triplet_test_dataset = TripletBlister_Dataset(
        ps1_ps2_test_dataset)  # Returns triplets of images
    kwargs = {'num_workers': NUM_WORKERS, 'pin_memory': True} if CUDA_AVAILABLE else {}
    triplet_test_loader = DataLoader(
        triplet_test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=True, **kwargs)

#%%
# train, evaluation functions
def train(model, optimizer, data_loader, loss_fn, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()
    model.cuda()

    with tqdm(total=total_samples, desc='Training') as t:
        loss_history_epoch = []
        for i, (item, target) in enumerate(data_loader):
            optimizer.zero_grad()
            
            anchor, pos, neg = item[0].cuda(), item[1].cuda(), item[2].cuda()
            anchor, pos, neg = anchor.half(), pos.half(), neg.half()
            
            anchor_output = model(anchor)
            pos_output = model(pos)
            neg_output = model(neg)
            
            # anchor_output, pos_output, neg_output = model(anchor, pos, neg)
            
            # assert if any input of loss_fn is nan
            if torch.isnan(anchor_output).any() or torch.isnan(pos_output).any() or torch.isnan(neg_output).any():
                print(anchor_output)
                print(pos_output)
                print(neg_output)
                print('\n')
                assert False and 'nan detected in loss_fn inputs'
                
            loss = loss_fn(anchor_output.float(), pos_output.float(), neg_output.float())
            
            loss.backward()
            optimizer.step()

            loss_history_epoch.append(loss.item())
            with torch.no_grad():
                t.set_postfix(loss='{:.4f}'.format(loss.item()))
            t.update(anchor.shape[0])
        loss_epoch = np.mean(loss_history_epoch)
        t.set_postfix(loss='{:.4f}'.format(loss_epoch))
    
    loss_history.append(loss_epoch)

def get_class_embed(model, triplet_dataset):
    prototype_dataset = Prototype_Dataset(triplet_dataset.blister_dataset, len(triplet_dataset.labels_set), list(range(48)))
    model.eval()
    model.cuda()
    
    cls_idx_2_embed = {}
    with tqdm(total=len(triplet_dataset.labels_set), desc='Getting class embeddings') as t:
        for cls_idx in triplet_dataset.labels_set:
            # embed_list = []
            imgs = torch.stack(prototype_dataset[cls_idx])
            
            with torch.no_grad():
                imgs = imgs.cuda()
                imgs = imgs.to(dtype=torch.float16)
                output = model(imgs)
                output = output.mean(dim=0)
            cls_idx_2_embed[cls_idx] = output
            t.update()

    return cls_idx_2_embed

def evaluate_classification(model: ImageTransformer, triplet_dataset: TripletBlister_Dataset, blister_loader: DataLoader, loss_history):
    model.eval()
    model.cuda()
    
    cls_idx_2_embed = get_class_embed(model, triplet_dataset)
    
    for _, (imgs, labels) in enumerate(blister_loader):
        imgs = imgs.half()
        imgs = imgs.cuda()
        
        with torch.no_grad():
            outputs = model(imgs)
            
        distances = torch.cdist(outputs, cls_idx_2_embed.values(), p=2)
        
        
        break
    
    return distances
            
def evaluate(model, data_loader, loss_fn, loss_history):
    total_samples = len(data_loader.dataset)
    model.eval()
    model.cuda()

    with tqdm(total=total_samples, desc='Evaluating') as t:
        loss_history_epoch = []
        with torch.no_grad():
            for i, (item, target) in enumerate(data_loader):
                anchor, pos, neg = item[0].cuda(), item[1].cuda(), item[2].cuda()
                anchor, pos, neg = anchor.half(), pos.half(), neg.half()
                
                anchor_output = model(anchor)
                pos_output = model(pos)
                neg_output = model(neg)
                loss = loss_fn(anchor_output.float(), pos_output.float(), neg_output.float())

                loss_history_epoch.append(loss.item())
                t.set_postfix(loss='{:.4f}'.format(loss.item()))
                t.update(anchor.shape[0])
                
        loss_epoch = np.mean(loss_history_epoch)
        t.set_postfix(loss='{:.4f}'.format(loss_epoch))
    
    loss_history.append(np.mean(loss_epoch))
    
#%%
# initialize model
if __name__ == '__main__':
    model = ImageTransformer(image_size=IMG_SIZE, patch_size=PATCH_SIZE, channels=3,
                dim=DIM, depth=DEPTH, heads=HEADS, mlp_dim=OUTPUT_DIM)
    # model = TripletNet(model)
    model = model.half()
    loss_fn = TripletLoss_fix()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

#%%
# run
if __name__ == '__main__':
    train_loss_history, test_loss_history = [], []
    
    if LOAD_MODEL:
        # load model if exists
        if os.path.exists(MODEL_PATH):
            print('Loading model from ' + MODEL_PATH)
            model.load_state_dict(torch.load(MODEL_PATH))
        else:
            print('Model not found at ' + MODEL_PATH)
    
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:{}/{}'.format(epoch, N_EPOCHS))
        train(model, optimizer, triplet_train_loader, loss_fn, train_loss_history)
        evaluate(model, triplet_test_loader, loss_fn, test_loss_history)
        
        # plot train_loss_history and test_loss_history in a figure
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label='train')
        plt.plot(test_loss_history, label='test')
        plt.legend()
        plt.show()
        
        # save model
        if epoch % SAVE_EPOCHS == 0:
            torch.save(model.state_dict(), MODEL_PATH)
            print('Saved model to ' + MODEL_PATH)

#%%