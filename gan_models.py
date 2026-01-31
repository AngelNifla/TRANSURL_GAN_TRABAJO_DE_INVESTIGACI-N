# gan_models.py
import torch
import torch.nn as nn

# Importa tu modelo original TransURL
# En tu Train.py lo usas así:
# from Model_MMA import Model, CharBertModel
from Model_MMA import Model


class TransURLDiscriminator(nn.Module):
    """
    Discriminador basado en TransURL:
    - Usa tu Model original como backbone.
    - Usa 'pooled' como vector de características.
    - Agrega:
        * cls_head: cabeza de clasificación benigno/malware sobre 'pooled'
        * adv_head: cabeza real/fake sobre 'pooled'
    """
    def __init__(self, hidden_dim=768, pretrained_weights_path=None):
        super().__init__()

        # Tu modelo TransURL original
        self.base_model = Model()  # igual que en tu Train.py

        # Si quieres cargar pesos preentrenados (TransURL baseline)
        if pretrained_weights_path is not None:
            state_dict = torch.load(pretrained_weights_path, map_location="cpu")
            self.base_model.load_state_dict(state_dict)

        self.hidden_dim = hidden_dim

        # Cabeza de clasificación: benigno/malware (2 clases)
        self.cls_head = nn.Linear(self.hidden_dim, 2)

        # Cabeza adversarial: real/fake (1 neurona)
        self.adv_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, inputs):
        """
        inputs: lista [x1, x2, x3] exactamente como en tu Train.py:
                [input_ids, input_types, input_masks]

        Retorna:
            cls_logits: logits benigno/malware (para cross_entropy)
            adv_logits: logits real/fake (nuevo)
            pooled: vector de características (h)
        """
        # Tu Model actual devuelve: outputs, pooled, y_pred
        outputs, pooled, _ = self.base_model(inputs)

        # Clasificación sobre el espacio de características
        cls_logits = self.cls_head(pooled)  # [B, 2]

        # Real/Fake sobre el mismo 'pooled'
        adv_logits = self.adv_head(pooled)  # [B, 1]

        return cls_logits, adv_logits, pooled


class ResidualBlock(nn.Module):
    """
    Bloque residual simple: Linear -> GELU -> Linear + skip
    Mantiene la dimensión fija (ej. 512).
    """
    def __init__(self, dim=512):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        return x + out  # conexión residual


class AdversarialGenerator(nn.Module):
    """
    Generador fuerte condicional y residual.

    - Recibe:
        h: vector de features reales (pooled) de TransURL [B, hidden_dim]
        y: etiqueta de clase (0 benign, 1 malware) [B]
        z: ruido gaussiano [B, latent_dim]
    - Produce:
        h_adv: features adversariales = h + δ_normalizada
        delta: perturbación aplicada
    """
    def __init__(self,
                 hidden_dim=768,
                 latent_dim=128,
                 num_classes=2,
                 class_emb_dim=16,
                 epsilon=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_emb_dim = class_emb_dim
        self.epsilon = epsilon  # tamaño máximo de perturbación (L2)

        # Embedding para la clase (0/1)
        self.class_emb = nn.Embedding(num_classes, class_emb_dim)

        # Entrada total: h (hidden_dim) + e_y (class_emb_dim) + z (latent_dim)
        in_dim = hidden_dim + class_emb_dim + latent_dim

        self.fc_in = nn.Linear(in_dim, 512)
        self.act_in = nn.GELU()

        # Más bloques residuales para más capacidad
        self.res_block1 = ResidualBlock(dim=512)
        self.res_block2 = ResidualBlock(dim=512)
        self.res_block3 = ResidualBlock(dim=512)

        # Proyección final a un vector delta en el espacio de h
        self.fc_out = nn.Linear(512, hidden_dim)

    def forward(self, h, y, z):
        """
        h: [B, hidden_dim]  (features reales de TransURL)
        y: [B]  (etiquetas enteras 0/1)
        z: [B, latent_dim] (ruido gaussiano)

        Retorna:
            h_adv: [B, hidden_dim]
            delta: [B, hidden_dim] perturbación aplicada
        """
        if y.dtype != torch.long:
            y = y.long()

        e_y = self.class_emb(y)  # [B, class_emb_dim]

        # Concatenamos h, clase embebida y ruido
        x = torch.cat([h, e_y, z], dim=-1)  # [B, in_dim]

        x = self.fc_in(x)
        x = self.act_in(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        delta_raw = self.fc_out(x)  # [B, hidden_dim]

        # Normalizamos la perturbación (L2) para controlar su tamaño
        norm = torch.norm(delta_raw, p=2, dim=-1, keepdim=True) + 1e-8
        delta = self.epsilon * delta_raw / norm  # [B, hidden_dim]

        h_adv = h + delta
        return h_adv, delta
