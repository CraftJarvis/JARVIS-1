import torch

# Define the model
mineclip_dim = 512
latent_dim = 256  # experiment with this
hidden_dim = 512  # experiment with this


# Define some helper functions to load the model.
def load_vae_model(vae_info, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """Load the VAE model from the given path."""
    # Extract the model parameters.
    mineclip_dim = vae_info['mineclip_dim']
    latent_dim = vae_info['latent_dim']
    hidden_dim = vae_info['hidden_dim']
    model_path = vae_info['model_path']

    model = TranslatorVAE(input_dim=mineclip_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# Define the model as a simple conditional MLP VAE.
class TranslatorVAE(torch.nn.Module):

    def __init__(self, input_dim=512, hidden_dim=256, latent_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, visual_embeddings, text_embeddings):
        """Encode the given visual and text embeddings into a latent vector."""
        # Concatenate the visual and text embeddings.
        x = torch.cat([visual_embeddings, text_embeddings], dim=1)
        # Encode the concatenated embeddings into a latent vector.
        return self.encoder(x)

    def sample(self, mu, logvar):
        """Sample a latent vector from the given mu and logvar."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent_vector, text_embeddings):
        """Decode the given latent vector and text embeddings into a visual embedding."""
        # Concatenate the latent vector and text embeddings.
        x = torch.cat([latent_vector, text_embeddings], dim=1)
        # Decode the concatenated embeddings into a visual embedding.
        return self.decoder(x)

    def forward(self, text_embeddings, deterministic=False):
        """Encode the given text embeddings into a latent vector and then decode it into a visual embedding."""
        # Use the prior as the mean and logvar.
        mu = torch.zeros(text_embeddings.shape[0], self.latent_dim).to(text_embeddings.device)
        logvar = torch.zeros(text_embeddings.shape[0], self.latent_dim).to(text_embeddings.device)

        # Sample a latent vector from the mu and logvar.
        if deterministic:
            latent_vector = mu
        else:
            latent_vector = self.sample(mu, logvar)

        # Decode the latent vector into a visual embedding.
        pred_visual_embeddings = self.decode(latent_vector, text_embeddings)

        return pred_visual_embeddings
