import argparse
from sklearn.utils import shuffle
import torch
import pickle
from sklearn.utils import shuffle
import numpy as np
import copy
from tqdm import tqdm

from jarvis.steveI.steveI_lib.data.text_alignment.vae import TranslatorVAE

MINECLIP_DIM = 512

def train_step(model, text_embeddings, visual_embeddings, optimizer, beta):
    """Train the model on the given text and visual embeddings."""
    with torch.cuda.amp.autocast():
        # Zero the gradients.
        optimizer.zero_grad()
        # Encode the text and visual embeddings into a latent vector.
        mu, logvar = model.encode(visual_embeddings, text_embeddings).chunk(2, dim=1)
        # Sample a latent vector from the mu and logvar.
        latent_vector = model.sample(mu, logvar)
        # Decode the latent vector and text embeddings into a visual embedding.
        pred_visual_embeddings = model.decode(latent_vector, text_embeddings)

        # Compute the ELBO loss for the VAE.
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = torch.nn.functional.mse_loss(pred_visual_embeddings, visual_embeddings)
        loss = beta * kl_loss + recon_loss
        # Backpropagate the loss.
        loss.backward()
        # Update the parameters.
        optimizer.step()
        return loss.item(), kl_loss.item(), recon_loss.item()

@torch.no_grad()
def val_step(model, text_embeddings, visual_embeddings, beta):
    with torch.cuda.amp.autocast():
        """Run a validation step on the given text and visual embeddings."""
        # Encode the text and visual embeddings into a latent vector.
        mu, logvar = model.encode(visual_embeddings, text_embeddings).chunk(2, dim=1)
        # Sample a latent vector from the mu and logvar.
        latent_vector = model.sample(mu, logvar)
        # Decode the latent vector and text embeddings into a visual embedding.
        pred_visual_embeddings = model.decode(latent_vector, text_embeddings)
        # Compute the ELBO loss for the VAE.
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = torch.nn.functional.mse_loss(pred_visual_embeddings, visual_embeddings)
        loss = beta * kl_loss + recon_loss
        return loss.item(), kl_loss.item(), recon_loss.item()

def main(args):
    print('Load data...')
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    all_text_embeddings = data['text_embeddings']
    all_visual_embeddings = data['visual_embeddings']

    print('Shuffling data...')
    all_text_embeddings, all_visual_embeddings = shuffle(all_text_embeddings, all_visual_embeddings)
    train_text_embeddings = all_text_embeddings[:-args.n_validation]
    train_visual_embeddings = all_visual_embeddings[:-args.n_validation]
    val_text_embeddings = all_text_embeddings[-args.n_validation:]
    val_visual_embeddings = all_visual_embeddings[-args.n_validation:]

    train_text_embeddings = np.array(train_text_embeddings).reshape(-1, 512)
    train_visual_embeddings = np.array(train_visual_embeddings).reshape(-1, 512)
    val_text_embeddings = np.array(val_text_embeddings).reshape(-1, 512)
    val_visual_embeddings = np.array(val_visual_embeddings).reshape(-1, 512)

    train_text_embeddings = torch.tensor(train_text_embeddings).float()
    train_visual_embeddings = torch.tensor(train_visual_embeddings).float()
    val_text_embeddings = torch.tensor(val_text_embeddings).float()
    val_visual_embeddings = torch.tensor(val_visual_embeddings).float()

    train_dataset = torch.utils.data.TensorDataset(train_text_embeddings, train_visual_embeddings)
    val_dataset = torch.utils.data.TensorDataset(val_text_embeddings, val_visual_embeddings)

    print('Loading model...')
    model = TranslatorVAE(MINECLIP_DIM, args.hidden_dim, args.latent_dim).cuda()
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Keep track of the best val loss and the epoch where this occurs
    best_val_loss = np.inf
    best_epoch = 0
    best_model = None

    for epoch in range(args.n_epochs):
        # Train the model
        model.train()
        train_losses = []
        train_kl_losses = []
        train_recon_losses = []
        for text_embeddings, visual_embeddings in tqdm(train_dataloader, desc=f'Epoch {epoch}'):
            loss, kl_loss, recon_loss = train_step(model, 
                                                   text_embeddings.cuda(), 
                                                   visual_embeddings.cuda(), 
                                                   optimizer, 
                                                   args.beta)
            train_losses.append(loss)
            train_kl_losses.append(kl_loss)
            train_recon_losses.append(recon_loss)
        # Validate the model
        model.eval()
        val_losses = []
        val_kl_losses = []
        val_recon_losses = []
        for text_embeddings, visual_embeddings in tqdm(val_dataloader, desc=f'Epoch {epoch}'):
            loss, kl_loss, recon_loss = val_step(model, 
                                                 text_embeddings.cuda(), 
                                                 visual_embeddings.cuda(),
                                                 args.beta)
            val_losses.append(loss)
            val_kl_losses.append(kl_loss)
            val_recon_losses.append(recon_loss)

        # Save the best val loss
        if np.mean(val_losses) < best_val_loss:
            best_val_loss = np.mean(val_losses)
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        # Print the epoch metrics
        print(f'Epoch {epoch} train loss: {np.mean(train_losses):.4f} val loss: {np.mean(val_losses):.4f}')
        print(f'   train kl loss: {np.mean(train_kl_losses):.4f} val kl loss: {np.mean(val_kl_losses):.4f}')
        print(f'   train recon loss: {np.mean(train_recon_losses):.4f} val recon loss: {np.mean(val_recon_losses):.4f}')

    # Print the best val loss and the epoch where this occurs
    print(f'Best val loss: {best_val_loss:.4f} at epoch {best_epoch}')
    # Unwrap the model
    model = model._orig_mod
    best_model = best_model._orig_mod

    model = best_model

    print('Saving model...')
    torch.save(model.state_dict(), args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/vae_data/data.pkl')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_validation', type=int, default=200)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--output_path', type=str, default='data/weights/vae/trained_vae.pt')
    parser.add_argument('--beta', type=float, default=1.0)

    args = parser.parse_args()
    main(args)