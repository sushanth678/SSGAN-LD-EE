from util import Trainer
from model import Generator, Discriminator
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

def mnist(batch_size=128):
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.MNIST('../data', train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST('../data', train=False,
                               transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

trainset,testset = mnist(batch_size = 128)

generator = Generator()
discriminator = Discriminator()

lr = 0.0002
gamma = 0.96
beta = (0.0,0.9)
step_size = 10
epochs = 100
generator_adam = optim.Adam(generator.parameters(), lr=lr, betas=beta)
generator_scheduler = optim.lr_scheduler.StepLR(generator_adam, step_size=step_size, gamma=gamma)
discriminator_adam = optim.Adam(generator.parameters(), lr=lr, betas=beta)
discriminator_scheduler = optim.lr_scheduler.StepLR(discriminator_adam, step_size=step_size, gamma=gamma)

training = Trainer(generator, generator_adam, generator_scheduler, 
					discriminator, discriminator_adam, discriminator_scheduler,
					gen_weight_rot_loss=0.5, dis_weight_rot_loss=1.0, use_cuda=False)

training.train(trainset,epochs)
torch.save(training.G.state_dict(), './gen_mnist.pt')
torch.save(training.D.state_dict(), './dis_mnist.pt')