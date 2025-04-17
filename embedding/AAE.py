# src/embedding/aae.py
"""对抗自编码器（AAE）用于嵌入降维。"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.cuda.amp import autocast, GradScaler
import logging
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """编码器，将高维嵌入映射到潜在空间。"""

    def __init__(self, input_dim: int = 3584, hidden_dim: int = 1024, latent_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Decoder(nn.Module):
    """解码器，从潜在空间重构嵌入。"""

    def __init__(self, input_dim: int = 3584, hidden_dim: int = 1024, latent_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.fc1(z))
        return self.fc2(z)


class Discriminator(nn.Module):
    """判别器，区分真实和虚假的潜在表示。"""

    def __init__(self, hidden_dim: int = 1024, latent_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.fc1(z))
        return self.fc2(z)


def load_data(job_path: str, user_path: str, batch_size: int, split_ratio: float = 0.8) -> tuple:
    """加载并分割嵌入数据为训练集和验证集。"""
    try:
        job_embeddings = torch.load(job_path)
        user_embeddings = torch.load(user_path)
        all_embeddings = torch.stack(list(job_embeddings.values()) + list(user_embeddings.values()))
        indices = torch.randperm(all_embeddings.size(0))
        shuffled_embeddings = all_embeddings[indices]

        train_size = int(split_ratio * shuffled_embeddings.size(0))
        train_subset, valid_subset = random_split(shuffled_embeddings,
                                                  [train_size, len(shuffled_embeddings) - train_size])

        train_embeddings = torch.stack([shuffled_embeddings[i] for i in train_subset.indices])
        valid_embeddings = torch.stack([shuffled_embeddings[i] for i in valid_subset.indices])

        train_loader = DataLoader(TensorDataset(train_embeddings), batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(valid_embeddings), batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader
    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        raise


def train_model(
        encoder: Encoder,
        decoder: Decoder,
        discriminator: Discriminator,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int,
        device: str,
        save_path: str,
        patience: int = 10
) -> None:
    """训练对抗自编码器。"""
    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()
    opt_encoder = optim.Adam(encoder.parameters(), lr=1e-5)
    opt_decoder = optim.Adam(decoder.parameters(), lr=1e-5)
    opt_discriminator = optim.Adam(discriminator.parameters(), lr=1e-5)

    best_loss = float('inf')
    no_improvement = 0

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        discriminator.train()
        total_recon_loss = 0
        total_batches = 0

        for x in train_loader:
            x = x[0].to(device)
            batch_size = x.size(0)
            total_batches += 1

            for _ in range(2):
                with autocast():
                    z = encoder(x)
                    x_recon = decoder(z)
                    recon_loss = F.mse_loss(x_recon, x, reduction='mean')

                    z_real = torch.randn(batch_size, encoder.fc2.out_features).to(device)
                    z_fake = encoder(x).detach()
                    d_real_logits = discriminator(z_real)
                    d_fake_logits = discriminator(z_fake)

                    real_labels = torch.ones_like(d_real_logits)
                    fake_labels = torch.zeros_like(d_fake_logits)
                    d_loss = criterion(d_real_logits, real_labels) + criterion(d_fake_logits, fake_labels)

                opt_discriminator.zero_grad()
                scaler.scale(d_loss).backward(retain_graph=True)
                scaler.step(opt_discriminator)
                scaler.update()

            with autocast():
                z_fake_4g = encoder(x)
                d_fake_logits_4g = discriminator(z_fake_4g)
                g_loss = criterion(d_fake_logits_4g, torch.ones_like(d_fake_logits_4g))
                total_loss = recon_loss + 0.1 * g_loss

            opt_encoder.zero_grad()
            opt_decoder.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(opt_encoder)
            scaler.step(opt_decoder)
            scaler.update()

            total_recon_loss += recon_loss.item()

        logger.info(f"Epoch {epoch + 1}/{epochs}, Recon Loss: {total_recon_loss / total_batches:.4f}")

        # 验证
        encoder.eval()
        decoder.eval()
        discriminator.eval()
        total_recon_loss = 0
        count = 0

        with torch.no_grad():
            for x in valid_loader:
                x = x[0].to(device)
                z = encoder(x)
                x_recon = decoder(z)
                recon_loss = F.mse_loss(x_recon, x, reduction='mean')
                total_recon_loss += recon_loss.item() * x.size(0)
                count += x.size(0)

        avg_recon_loss = total_recon_loss / count
        logger.info(f"验证 - 重构损失: {avg_recon_loss:.4f}")

        if avg_recon_loss < best_loss:
            best_loss = avg_recon_loss
            no_improvement = 0
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
            }, save_path)
            logger.info(f"最佳模型在第 {epoch + 1} 轮保存，验证损失为 {best_loss:.4f}")
        else:
            no_improvement += 1

        if no_improvement >= patience:
            logger.info("提前停止触发。")
            break


def reduce_embeddings(
        encoder: Encoder,
        job_path: str,
        user_path: str,
        device: str,
        save_job_path: str,
        save_user_path: str
) -> None:
    """使用训练好的编码器降维嵌入。"""
    encoder.eval()
    job_embeddings = torch.load(job_path)
    user_embeddings = torch.load(user_path)

    reduced_job_embeddings = {
        k: encoder(v.unsqueeze(0).to(device)).cpu().squeeze(0)
        for k, v in job_embeddings.items()
    }
    reduced_user_embeddings = {
        k: encoder(v.unsqueeze(0).to(device)).cpu().squeeze(0)
        for k, v in user_embeddings.items()
    }

    torch.save(reduced_job_embeddings, save_job_path)
    torch.save(reduced_user_embeddings, save_user_path)
    logger.info(f"降维后的嵌入已保存到 {save_job_path} 和 {save_user_path}")


def main(args):
    """训练AAE并降维嵌入的主函数。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader = load_data(args.job_path, args.user_path, args.batch_size)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    discriminator = Discriminator().to(device)

    train_model(
        encoder, decoder, discriminator,
        train_loader, valid_loader, args.epochs, device, args.model_path
    )

    reduce_embeddings(
        encoder, args.job_path, args.user_path, device, args.save_job_path, args.save_user_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练AAE进行嵌入降维")
    parser.add_argument("--job_path", type=str, required=True, help="职位嵌入路径")
    parser.add_argument("--user_path", type=str, required=True, help="用户嵌入路径")
    parser.add_argument("--model_path", type=str, required=True, help="保存AAE模型的路径")
    parser.add_argument("--save_job_path", type=str, required=True, help="保存降维后职位嵌入的路径")
    parser.add_argument("--save_user_path", type=str, required=True, help="保存降维后用户嵌入的路径")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
    args = parser.parse_args()
    main(args)