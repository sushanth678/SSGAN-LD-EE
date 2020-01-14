import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torch.nn.functional as F
import imageio
import numpy as np
from torch.autograd import Variable
from torch.autograd import grad

class Trainer():
    def __init__(self, generator, generator_adam, generator_scheduler, discriminator,
                 discriminator_adam, discriminator_scheduler, gen_weight_rot_loss,
                 dis_weight_rot_loss, print_after=50, weight_gradientpenalty=10, gen_per_dis_iter=5,
                 use_cuda=False):

        self.G = generator
        self.G_opt = generator_adam
        self.G_sch = generator_scheduler
        self.D = discriminator
        self.D_opt = discriminator_adam
        self.D_sch = discriminator_scheduler
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()
        self.dis_weight_rot_loss = dis_weight_rot_loss
        self.gen_weight_rot_loss = gen_weight_rot_loss
        self.use_cuda = use_cuda
        self.weight_gradientpenalty = weight_gradientpenalty
        self.iterations = 0
        self.G_loss = []
        self.D_loss = []
        self.GP = []
        self.GN = []
        self.gen_per_dis_iter = gen_per_dis_iter
        self.print_after = print_after

    def sample(self, sample_size):

        generated = self.noise_generator(sample_size)
        generated_tonumpy = generated.data.cpu().numpy()
        one_channel = generated_tonumpy[:,0,:,:]

        return one_channel

    def noise_generator(self, sample_size):

        if self.use_cuda:
            samples = Variable(self.G.noise_sampling(sample_size)).cuda()
        else:
            samples = Variable(self.G.noise_sampling(sample_size))
        noisy_data = self.G(samples)
        
        return noisy_data

    def train_generator(self, fake_data, batch_size):

        self.G_opt.zero_grad()
        rotation = torch.zeros(4*batch_size,)
        if self.use_cuda:
            rotation = rotation.cuda()
        for i in range(4*batch_size):
            if i < batch_size:
                rotation[i] = 0
            elif i < 2*batch_size:
                rotation[i] = 1
            elif i < 3*batch_size:
                rotation[i] = 2
            else:
                rotation[i] = 3

        trash, g_fake_pro_logits, g_fake_rot_logits, g_fake_rot_prob = self.D(fake_data)
        g_loss = - torch.sum(g_fake_pro_logits)        
        rotation = F.one_hot(rotation.to(torch.int64), 4)
        rotation = rotation.float()
        g_fake_loss = torch.sum(F.binary_cross_entropy_with_logits(
                                                input = g_fake_rot_logits, 
                                                target = rotation))

        g_loss += self.gen_weight_rot_loss*g_fake_loss
        g_loss.backward(retain_graph=True)
        self.G_loss.append(g_loss.data)
        self.G_opt.step()

    def train_discriminator(self, real_data, fake_data, batch_size):
        
        self.D_opt.zero_grad()
        rotation = torch.zeros(4*batch_size)
        if self.use_cuda:
            rotation = rotation.cuda()
        
        for i in range(4*batch_size):
            if i < batch_size:
                rotation[i] = 0
            elif i < 2*batch_size:
                rotation[i] = 1
            elif i < 3*batch_size:
                rotation[i] = 2
            else:
                rotation[i] = 3

        tensor = Variable(real_data)
        if self.use_cuda:
            tensor = data.cuda()
        _, d_real_pro_logits, d_real_rot_logits, d_real_rot_prob = self.D(tensor)
        _, g_fake_pro_logits, g_fake_rot_logits, g_fake_rot_prob = self.D(fake_data)
        gp = self.gp(tensor, fake_data)
        self.GP.append(gp.data)

        d_loss = torch.sum(g_fake_pro_logits) - torch.sum(d_real_pro_logits)
        d_loss += gp      
        rotation = F.one_hot(rotation.to(torch.int64), 4)
        rotation = rotation.float()
        d_real_loss = torch.sum(F.binary_cross_entropy_with_logits(
                                            input = d_real_rot_logits,
                                            target = rotation))
        d_loss += self.dis_weight_rot_loss * d_real_loss
        d_loss.backward(retain_graph=True)
        self.D_opt.step()
        self.D_loss.append(d_loss.data)

    def gp(self, real, fake):

        batch_size = real.size()[0]
        alpha = torch.rand(batch_size, 1, 1, 1).expand_as(real)
        if self.use_cuda:
            alpha = alpha.cuda()
        stretch = Variable(alpha * real.data + (1 - alpha) * fake.data, requires_grad=True)
        if self.use_cuda:
            stretch = stretch.cuda()
        a, stretch_prob, garbage, trash = self.D(stretch)
        op = torch.ones(stretch_prob.size())
        if self.use_cuda:
            op = op.cuda()
        gradients = grad(inputs=stretch, outputs=stretch_prob,
                            grad_outputs=op,create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)
        self.GN.append(gradients.norm(2, dim=1).sum().data)
        norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gp_return = self.weight_gradientpenalty * ((norm - 1) ** 2).mean()
        return gp_return

    def train_one_epoch(self, dataloader):

        for i, data in enumerate(dataloader):

            batch = data[0]
            batch_size = batch.size()[0]
            generated = self.noise_generator(batch_size)
            
            fake_img = generated
            fake_img_90 = fake_img.transpose(2,3)
            fake_img_180 = fake_img.flip(2,3)
            fake_img_270 = fake_img.transpose(2,3).flip(2,3)
            generated = torch.cat((fake_img, fake_img_90, fake_img_180, fake_img_270),0)

            real_img = batch
            real_img_90 = real_img.transpose(2,3)
            real_img_180 = real_img.flip(2,3)
            real_img_270 = real_img.transpose(2,3).flip(2,3)
            batch = torch.cat((real_img,real_img_90,real_img_180,real_img_270),0)

            self.iterations += 1
            self.train_discriminator(batch, generated, batch_size)
            if self.iterations % self.gen_per_dis_iter == 0:
                self.train_generator(generated, batch_size)

            if i % self.print_after == 0:
                print("Iteration {}".format(i + 1))
                print("Gradient norm: {}".format(self.GN[-1]))

    def train(self, data, epochs):

        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self.train_one_epoch(data)
            self.G_sch.step()
            self.D_sch.step()