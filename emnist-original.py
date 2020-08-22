import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import copy

from torchvision.datasets import FashionMNIST, EMNIST
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Deep Dictionary Configurations
input_dim = 784  # the input dimensions to be expected
dd_layer_config = [784//2]  # the layer configuration for the deep dictionary
sparse_cff = 1e-1  # regularization to enusure sparseness in the dictionary representation
epoch_per_level = 15  # the number of epochs to train for each layer of deep dictionary

# MLP Configurations
batch_size_train = 500    # the batch size of the MLP model (optimized via Adam)
batch_size_valid = 500
epoch_mlp = 25              # the number of epochs to train the MLP for
num_classes = 47  # the number of classes for classification (10 for MNIST)
mlp_lr = 5e-3  # the learning rate for the Adam optimizer to optimize the MLP model


# prepare data loaders
mnist_train_data = EMNIST('./data/', split='balanced', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
train_data, valid_data = torch.utils.data.random_split(mnist_train_data, [90240, 22560], generator=torch.Generator().manual_seed(0))

train_loader_dd = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=False, pin_memory=True)
train_loader_mlp = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True, pin_memory=True)

valid_loader_mlp = torch.utils.data.DataLoader(valid_data, batch_size=batch_size_valid, shuffle=True, pin_memory=True)

test_data = EMNIST('./data/', split='balanced', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_loader_mlp = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True, pin_memory=True)

# Function Class
class Identity:
    @staticmethod
    def forward(x):
        return x

    @staticmethod
    def inverse(x):
        return x


class ELUInv:

    alpha = 1.5

    @staticmethod
    def forward(x):
        return (x > 0).float()*x + (x <= 0).float()*torch.log((x/ELUInv.alpha)+1)

    @staticmethod
    def inverse(x):
        return (x > 0).float() * x + (x <= 0).float() * ELUInv.alpha * (torch.exp(x) - 1)


class TanhInv:

    alpha = 10

    @staticmethod
    def forward(x):
        return TanhInv.alpha*torch.atanh(x)

    @staticmethod
    def inverse(x):
        return torch.tanh(x/TanhInv.alpha)

class ElliotSig:

    alpha = 2.5

    @staticmethod
    def forward(x):
        return ElliotSig.alpha*x/(1 + torch.abs(x))

    @staticmethod
    def inverse(x):
        return (x >= 0).float()*(x/(ElliotSig.alpha-x)) + (x < 0).float()*(x/(ElliotSig.alpha+x))


class DeepDictionary():

    def __init__(self, input_dim, **kwargs):
        """
        :param input_dim: the input dimension to be expected
        :param layer_config: the configuration for the layer dimensions of each dictionary layer
        :param activation: the activation function (default: None)
        :param sparseness_coeff: the sparseness coefficient
        """

        self.input_dim = input_dim

        self.layer_config = kwargs.get('layer_config', [self.input_dim, self.input_dim//2])  # list of layer dimensions
        if self.layer_config[0] != self.input_dim:  # in case where the layer config is given as hidden dimensions only
            self.layer_config = [self.input_dim] + self.layer_config
        assert len(self.layer_config) >= 2, "Error in specifying layer configuration, not enough layers"

        self.activation = kwargs.get('activation', Identity)    # default implies linear
        self.spar_cff = kwargs.get('sparseness_coeff', 0)   # default sparseness coefficient

        # construct dictionary
        self.dictionary_layers = [torch.rand((self.layer_config[i+1], self.layer_config[i]), requires_grad=False).to(device)
                                  for i in range(len(self.layer_config)-1)]

    def eval_layer(self, x, layer, concat_prev=False):
        """
        :param x: input of dimension (batch x dim)
        :param layer: the layer to be evaluated at (value between 0 and len(self.dictionary_layers)-1)
        :return: (Z_i, Z_i-1) 'Z' at layer 'i' and 'i-1' (batch x dim[layer])
        """

        assert layer in range(0, len(self.dictionary_layers)), "Error with layer specified (out of range)"

        # compute initial layer z_0
        d_i = self.dictionary_layers[0]
        z_i_prev = x
        z_i = z_i_prev@d_i.T@torch.inverse(d_i@d_i.T)  # first obtain Z_0

        if concat_prev:
            concat_z = z_i.clone()

        # process intermediate layer (if specified)
        for i in range(1, layer+1):  # iterate through next few layers to compute 'z_i' (if needed)
            d_i = self.dictionary_layers[i]  # obtain dictionary for this layer
            z_i_prev = z_i  # make a copy of previous z_i
            z_i_prev_ia = self.activation.inverse(z_i_prev)  # compute the inverse activated z_i_prev

            if i == len(self.dictionary_layers) - 1:  # if is last layer of deep model
                z_i = z_i_prev_ia@d_i.T@torch.inverse(d_i@d_i.T + self.spar_cff*torch.eye((len(d_i)), device=device))   # enforce sparseness
            else:
                z_i = z_i_prev_ia@d_i.T@torch.inverse(d_i@d_i.T)    # otherwise, treat as regular

            if concat_prev:
                concat_z = torch.cat([concat_z, z_i], dim=-1)

        if concat_prev:
            return z_i, z_i_prev, concat_z
        return z_i, z_i_prev


    def optimize_layer(self, x, layer):
        """
        :param x: input of dimension (batch x dim)
        :param layer: the layer to be trained (value between 0 and len(self.dictionary_layers)-1)
        :return: None
        """

        # only optimize specified layer (previous layers assumed constant during this specific layer optimization)

        assert layer in range(0, len(self.dictionary_layers)), "Error with layer specified (out of range)"

        z_layer, z_layer_prev = self.eval_layer(x, layer)  # obtain z_layer for the specified layer

        if layer == 0:  # layer '0' has no activations
            d_layer = torch.inverse(z_layer.T @ z_layer) @ z_layer.T @ z_layer_prev  # get optimal dictionary
        else:
            z_layer_prev_ia = self.activation.inverse(z_layer_prev)  # compute the inverse activated z_layer_prev
            d_layer = torch.inverse(z_layer.T@z_layer)@z_layer.T@z_layer_prev_ia

        self.dictionary_layers[layer] = d_layer  # update model dictionary

    def layer_reconstruction(self, x, layer):
        """
        Performs layer reconstruction of the previous latent 'z'
        :param x: input of dimension (batch x dim)
        :param layer: the layer to perform the reconstruction on
        :return: a reconstruction of 'x' based on learned dictionaries
        """

        assert layer in range(0, len(self.dictionary_layers)), "Error with layer specified (out of range)"

        z_layer, z_layer_prev = self.eval_layer(x, layer)
        layer_dict = self.dictionary_layers[layer]

        if layer == 0:
            z_layer_rec = z_layer @ layer_dict  # if first layer, linear activation
        else:
            z_layer_rec = self.activation.forward(z_layer@layer_dict)  # otherwise, non-linear activation

        return z_layer_rec, z_layer_prev

    def reconstruction(self, x):
        """
        Performs total reconstruction of the input image
        :param x: input of dimension (batch x dim)
        :return: a reconstruction of 'x' based on learned dictionaries
        """

        z_layer, _ = self.eval_layer(x, len(self.dictionary_layers)-1)

        # intermediate layers
        for dicts in self.dictionary_layers[:0:-1]:  # going reverse order, skip dictionary[0]
            z_layer = self.activation.forward(z_layer@dicts)

        # layer 0
        x_rec = z_layer@self.dictionary_layers[0]

        return x_rec


# define the models of the Deep Dictionary and MLP models

dd = DeepDictionary(input_dim=input_dim, layer_config=dd_layer_config,
                          activation=Identity, sparseness_coeff=sparse_cff)

mlp_input_dim = dd_layer_config[-1]

dd_mlp = nn.Sequential(
    nn.Linear(mlp_input_dim, mlp_input_dim//2),
    nn.Sigmoid(),
    nn.Linear(mlp_input_dim//2, mlp_input_dim//4),
    nn.Sigmoid(),
    nn.Linear(mlp_input_dim//4, num_classes)).to(device)

mlp_opt = torch.optim.Adam(dd_mlp.parameters(), lr=mlp_lr)
opt_schd = torch.optim.lr_scheduler.MultiStepLR(mlp_opt, [35, 50], gamma=0.25)

# begin model trainings
print('BEGIN TRAINING THE DEEP DICTIONARY MODEL')
for layer_i in range(len(dd.dictionary_layers)):
    for epoch in range(epoch_per_level):
        for batch_i, (img_dd, labels) in enumerate(train_loader_dd):
            # img_dd is batch of images - (batch x 1 x 28 x 28)
            # labels is batch of labels - (batch)

            img_dd = img_dd.to(device)
            batch_size, _, img_h, img_w = img_dd.shape
            img_dd = img_dd.view(batch_size, -1)

            # optimization
            dd.optimize_layer(img_dd, layer_i)
            img_rec = dd.reconstruction(img_dd)
            z_rec, z_prev = dd.layer_reconstruction(img_dd, layer_i)

            # eval
            total_loss = torch.sum((img_dd - img_rec) ** 2) / batch_size
            lat_rec_loss = torch.sum((z_prev - z_rec) ** 2) / batch_size
            print(f'Layer: {layer_i} | Epoch:{epoch} - Batch {batch_i} - '
                  f'| Total Loss: {total_loss:.4f} | Latent Loss: {lat_rec_loss:.4f}')


print('BEGIN TRAINING THE MLP MODEL')

best_metric, best_model_state_dict = 0, None

for epoch in range(epoch_mlp):

    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []

    # Training MLP
    dd_mlp.train()
    for batch_i, (img, labels) in enumerate(train_loader_mlp):
        mlp_opt.zero_grad()

        # compute latent variables
        with torch.no_grad():
            img = img.to(device)
            labels = labels.to(device)
            batch_size, _, img_h, img_w = img.shape
            img = img.view(batch_size, -1)

            final_layer_latent, _ = dd.eval_layer(img, len(dd.dictionary_layers) - 1)
            final_layer_latent = final_layer_latent.detach()

        # prediction and compute loss
        class_logits = dd_mlp(final_layer_latent)  # class_logits : (batch size x num_classes)
        ce_loss = nn.functional.cross_entropy(class_logits, labels)

        # optimization
        ce_loss.backward()
        mlp_opt.step()

        # eval
        class_probs = nn.functional.softmax(class_logits, dim=-1)
        class_pred = class_probs.argmax(dim=-1)
        acc = (class_pred == labels).float().mean().item()

        train_loss.append(ce_loss.item())
        train_acc.append(acc)

    # Evaluation MLP
    dd_mlp.eval()
    for batch_i, (img, labels) in enumerate(valid_loader_mlp):
        # compute latent variables
        with torch.no_grad():
            img = img.to(device)
            labels = labels.to(device)
            batch_size, _, img_h, img_w = img.shape
            img = img.view(batch_size, -1)

            final_layer_latent, _ = dd.eval_layer(img, len(dd.dictionary_layers) - 1)
            final_layer_latent = final_layer_latent.detach()


        # prediction and compute loss
        class_logits = dd_mlp(final_layer_latent)  # class_logits : (batch size x num_classes)
        ce_loss = nn.functional.cross_entropy(class_logits, labels)

        # eval
        class_probs = nn.functional.softmax(class_logits, dim=-1)
        class_pred = class_probs.argmax(dim=-1)
        acc = (class_pred == labels).float().mean().item()

        valid_loss.append(ce_loss.item())
        valid_acc.append(acc)

    epoch_total_acc_train = sum(train_acc)/len(train_acc)
    epoch_total_loss_train = sum(train_loss)/len(train_loss)
    epoch_total_acc_valid = sum(valid_acc) / len(valid_acc)
    epoch_total_loss_valid = sum(valid_loss) / len(valid_loss)

    print(f'---------------------------- Epoch:{epoch} ----------------------------------------')
    print(f'[TRAIN] | Loss: {epoch_total_loss_train:.4f} | ACC: {epoch_total_acc_train:.4f}')
    print(f'[VALID] | Loss: {epoch_total_loss_valid:.4f} | ACC: {epoch_total_acc_valid:.4f}')

    # record the best metric
    if epoch_total_acc_valid > best_metric:
        best_model_state_dict = copy.deepcopy(dd_mlp.state_dict())
        best_metric = epoch_total_acc_valid

    opt_schd.step()


# Final Evaluation on Test Set
# Evaluation MLP
dd_mlp.load_state_dict(best_model_state_dict)
dd_mlp.eval()
test_loss, test_correct = 0, 0

for batch_i, (img, labels) in enumerate(test_loader_mlp):
    # compute latent variables
    with torch.no_grad():
        img = img.to(device)
        labels = labels.to(device)
        batch_size, _, img_h, img_w = img.shape
        img = img.view(batch_size, -1)

        final_layer_latent, _ = dd.eval_layer(img, len(dd.dictionary_layers) - 1)
        final_layer_latent = final_layer_latent.detach()


    # prediction and compute loss
    class_logits = dd_mlp(final_layer_latent)  # class_logits : (batch size x num_classes)
    ce_loss = nn.functional.cross_entropy(class_logits, labels, reduction='sum')
    test_loss += ce_loss.item()

    # eval
    class_probs = nn.functional.softmax(class_logits, dim=-1)
    class_pred = class_probs.argmax(dim=-1)
    num_correct = (class_pred == labels).float().sum().item()
    test_correct += num_correct


test_loss_avg = test_loss/len(test_loader_mlp.dataset)
test_acc = test_correct/len(test_loader_mlp.dataset)

print(f'---------------------------- FINAL TEST RESULTS ----------------------------------------')
print(f'[TEST] | Loss: {test_loss_avg:.4f} | ACC: {test_acc:.4f}')





