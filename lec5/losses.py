import torch
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature):
    """NT-Xent on one positive pair (z1[i], z2[i]) per image.

    Both views are stacked into one matrix. Each row is pulled toward its
    partner and pushed from every other row. The vectors are L2-normalized
    first, so the dot product is cosine similarity.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    out = torch.cat([z1, z2], dim=0)  # [2B, D]
    n_samples = out.shape[0]

    # Full similarity matrix, [2B, 2B].
    sim = torch.exp(torch.mm(out, out.t()) / temperature)
    # Drop the diagonal. A vector is not a negative for itself.
    # The partner sits off the diagonal, so it stays in this sum.
    mask = ~torch.eye(n_samples, device=sim.device, dtype=torch.bool)
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # Positive similarity, counted both ways: (z1, z2) and (z2, z1).
    pos = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    return -torch.log(pos / neg).mean()


def triplet_loss(z1, z2, margin):
    """Triplet loss. z1[i] is the anchor, z2[i] is its positive.

    Every other row of z2 is a negative, another image in the batch.
    For each such triple the loss is

        max(0, d(anchor, positive) - d(anchor, negative) + margin)

    d is squared Euclidean distance. The vectors are L2-normalized first,
    so d(a, b) = 2 - 2 cos(a, b).
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if z1.shape[0] < 2:
        raise ValueError("triplet loss needs at least two images in the batch")

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # Pairwise distances, [B, B]. Entry (i, j) is d(z1[i], z2[j]).
    # The diagonal is the positive pair.
    dist = (2 - 2 * torch.mm(z1, z2.t())).clamp(min=0)
    pos = dist.diag().unsqueeze(1)  # [B, 1]

    loss = F.relu(pos - dist + margin)
    # The diagonal is the positive against itself, not a negative.
    eye = torch.eye(dist.shape[0], device=dist.device, dtype=torch.bool)
    loss = loss.masked_fill(eye, 0)
    n_negatives = dist.shape[0] - 1
    return loss.sum() / (dist.shape[0] * n_negatives)
