import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img

def train(D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=250, 
              batch_size=128, noise_size=100, num_epochs=10, train_loader=None, device=None):
    iter_count = 0
    for epoch in range(num_epochs):
        print('EPOCH: ', (epoch+1))
        for x, _ in train_loader:
            _, input_channels, img_size, _ = x.shape
            
            real_images = preprocess_img(x).to(device='cuda:0')  # normalize
            
            # Store discriminator loss output, generator loss output, and fake image output
            # in these variables for logging and visualization below
            d_error = None
            g_error = None
            fake_images = None
            # Step 1
            D_solver.zero_grad()
            # def sample_noise(batch_size, dim):
            FakeData = sample_noise(batch_size, noise_size).to(device='cuda:0')
            fake_images = G(FakeData).detach()
            real_logits = D(real_images)
            fake_logits = D(fake_images.view(batch_size, input_channels, img_size, img_size))
            d_error = discriminator_loss(real_logits, fake_logits)
            d_error.backward()
            D_solver.step()

            # Step 2
            G_solver.zero_grad()
            FakeData = sample_noise(batch_size, noise_size).to(device='cuda:0')
            fake_images = G(FakeData)
            fake_logits2 = D(fake_images.view(batch_size, input_channels, img_size, img_size))
            g_error = generator_loss(fake_logits2)
            g_error.backward()
            G_solver.step()
            # Logging and output visualization
            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_error.item(),g_error.item()))
                disp_fake_images = deprocess_img(fake_images.data)  # denormalize
                imgs_numpy = (disp_fake_images).cpu().numpy()
                show_images(imgs_numpy[0:16], color=input_channels!=1)
                plt.show()
                print()
            iter_count += 1