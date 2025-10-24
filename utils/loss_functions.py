"""
Loss functions implementations used in the optimization of DLP.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from collections import namedtuple
import os
import hashlib
import requests
from PIL import Image
from tqdm import tqdm

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}


# functions
def batch_pairwise_kl(mu_x, logvar_x, mu_y, logvar_y, reverse_kl=False):
    """
    Calculate batch-wise KL-divergence
    mu_x, logvar_x: [batch_size, n_x, points_dim]
    mu_y, logvar_y: [batch_size, n_y, points_dim]
    kl = -0.5 * Σ_points_dim (1 + logvar_x - logvar_y - exp(logvar_x)/exp(logvar_y)
                    - ((mu_x - mu_y) ** 2)/exp(logvar_y))
    """
    if reverse_kl:
        mu_a, logvar_a = mu_y, logvar_y
        mu_b, logvar_b = mu_x, logvar_x
    else:
        mu_a, logvar_a = mu_x, logvar_x
        mu_b, logvar_b = mu_y, logvar_y
    bs, n_a, points_dim = mu_a.size()
    _, n_b, _ = mu_b.size()
    logvar_aa = logvar_a.unsqueeze(2).expand(-1, -1, n_b, -1)  # [batch_size, n_a, n_b, points_dim]
    logvar_bb = logvar_b.unsqueeze(1).expand(-1, n_a, -1, -1)  # [batch_size, n_a, n_b, points_dim]
    mu_aa = mu_a.unsqueeze(2).expand(-1, -1, n_b, -1)  # [batch_size, n_a, n_b, points_dim]
    mu_bb = mu_b.unsqueeze(1).expand(-1, n_a, -1, -1)  # [batch_size, n_a, n_b, points_dim]
    p_kl = -0.5 * (1 + logvar_aa - logvar_bb - logvar_aa.exp() / logvar_bb.exp()
                   - ((mu_aa - mu_bb) ** 2) / logvar_bb.exp()).sum(-1)  # [batch_size, n_x, n_y]
    return p_kl


def batch_pairwise_dist(x, y, metric='l2'):
    assert metric in ['l2', 'l2_simple', 'l1', 'cosine'], f'metric {metric} unrecognized'
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    if metric == 'cosine':
        dist_func = torch.nn.functional.cosine_similarity
        P = -dist_func(x.unsqueeze(2), y.unsqueeze(1), dim=-1, eps=1e-8)
    elif metric == 'l1':
        P = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(-1)
    elif metric == 'l2_simple':
        P = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    else:
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x, device=x.device)
        diag_ind_y = torch.arange(0, num_points_y, device=y.device)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
    return P


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """

    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum', balance=0.5):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :param balance: balancing coefficient between posterior and prior
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    if balance == 0.5:
        # kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / (torch.exp(logvar_o) + eps) - (mu - mu_o).pow(2) / (
        #             torch.exp(logvar_o) + eps)).sum(-1)
        kl = -0.5 * (1 + logvar - logvar_o - torch.exp(logvar - logvar_o) - (mu - mu_o).pow(2) * torch.exp(
            -logvar_o)).sum(-1)
    else:
        # detach post
        mu_post = mu.detach()
        logvar_post = logvar.detach()
        mu_prior = mu_o
        logvar_prior = logvar_o
        # kl_a = -0.5 * (1 + logvar_post - logvar_prior - logvar_post.exp() / (torch.exp(logvar_prior) + eps) - (
        #         mu_post - mu_prior).pow(2) / (torch.exp(logvar_prior) + eps)).sum(-1)
        kl_a = -0.5 * (1 + logvar_post - logvar_prior - torch.exp(logvar_post - logvar_prior) - (
                mu_post - mu_prior).pow(2) * torch.exp(-logvar_prior)).sum(-1)
        # detach prior
        mu_post = mu
        logvar_post = logvar
        mu_prior = mu_o.detach()
        logvar_prior = logvar_o.detach()
        # kl_b = -0.5 * (1 + logvar_post - logvar_prior - logvar_post.exp() / (torch.exp(logvar_prior) + eps) - (
        #         mu_post - mu_prior).pow(2) / (torch.exp(logvar_prior) + eps)).sum(-1)
        kl_b = -0.5 * (1 + logvar_post - logvar_prior - torch.exp(logvar_post - logvar_prior) - (
                mu_post - mu_prior).pow(2) * torch.exp(-logvar_prior)).sum(-1)
        kl = (1 - balance) * kl_a + balance * kl_b
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def calc_kl_bern(post_prob, prior_prob, eps=1e-15, reduce='none'):
    """
    Compute kl divergence of Bernoulli variable
    :param post_prob [batch_size, 1], in [0,1]
    :param prior_prob [batch_size, 1], in [0,1]
    :return: kl divergence, (B, ...)
    """
    kl = post_prob * (torch.log(post_prob + eps) - torch.log(prior_prob + eps)) + (1 - post_prob) * (
            torch.log(1 - post_prob + eps) - torch.log(1 - prior_prob + eps))
    if reduce == 'sum':
        kl = kl.sum()
    elif reduce == 'mean':
        kl = kl.mean()
    else:
        kl = kl.squeeze(-1)
    return kl


def log_beta_function(alpha, beta, eps: float = 1e-5):
    """
    B(alpha, beta) = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
    logB = loggamma(alpha) + loggamma(beta) - loggamaa(alpha + beta)
    """
    # return torch.special.gammaln(alpha) + torch.special.gammaln(beta) - torch.special.gammaln(alpha + beta)
    return torch.lgamma(alpha + eps) + torch.lgamma(beta + eps) - torch.lgamma(alpha + beta + eps)


def calc_kl_beta_dist(alpha_post, beta_post, alpha_prior, beta_prior, reduce: str = 'none', eps: float = 1e-5,
                      balance: float = 0.5):
    """
    Compute kl divergence of Beta variable
    https://en.wikipedia.org/wiki/Beta_distribution
    :param alpha_post, beta_post [batch_size, 1]
    :param alpha_prior,  beta_prior  [batch_size, 1]
    :param balance kl balance between posterior and prior
    :return: kl divergence, (B, ...)
    """
    if balance == 0.5:
        log_bettas = log_beta_function(alpha_prior, beta_prior) - log_beta_function(alpha_post, beta_post)
        alpha = (alpha_post - alpha_prior) * torch.digamma(alpha_post + eps)
        beta = (beta_post - beta_prior) * torch.digamma(beta_post + eps)
        alpha_beta = (alpha_prior - alpha_post + beta_prior - beta_post) * torch.digamma(alpha_post + beta_post + eps)
        kl = log_bettas + alpha + beta + alpha_beta
    else:
        # detach post
        log_bettas = log_beta_function(alpha_prior, beta_prior) - log_beta_function(alpha_post.detach(),
                                                                                    beta_post.detach())
        alpha = (alpha_post - alpha_prior) * torch.digamma(alpha_post.detach() + eps)
        beta = (beta_post.detach() - beta_prior) * torch.digamma(beta_post.detach() + eps)
        alpha_beta = (alpha_prior - alpha_post.detach() + beta_prior - beta_post.detach()) * torch.digamma(
            alpha_post.detach() + beta_post.detach() + eps)
        kl_a = log_bettas + alpha + beta + alpha_beta

        # detach prior
        log_bettas = log_beta_function(alpha_prior.detach(), beta_prior.detach()) - log_beta_function(alpha_post,
                                                                                                      beta_post)
        alpha = (alpha_post - alpha_prior.detach()) * torch.digamma(alpha_post + eps)
        beta = (beta_post - beta_prior.detach()) * torch.digamma(beta_post + eps)
        alpha_beta = (alpha_prior.detach() - alpha_post + beta_prior.detach() - beta_post) * torch.digamma(
            alpha_post + beta_post + eps)
        kl_b = log_bettas + alpha + beta + alpha_beta
        kl = (1 - balance) * kl_a + balance * kl_b
    if reduce == 'sum':
        kl = kl.sum()
    elif reduce == 'mean':
        kl = kl.mean()
    else:
        kl = kl.squeeze(-1)
    return kl


def calc_kl_categorical(logits_post, logits_prior, num_classes: int = 4, reduce: str = 'none', balance: float = 0.5):
    """
    Compute kl divergence of categorical variable
    :param logits_post, beta_post [batch_size, num_categories * num_classes]
    :param logits_prior,  beta_prior  [batch_size, num_categories * num_classes]
    :param balance kl balance between posterior and prior
    :return: kl divergence, (B, ...)
    """
    orig_shape = logits_post.shape
    logits_post = logits_post.view(-1, num_classes)
    logits_prior = logits_prior.view(-1, num_classes)
    post_logprobs = torch.log_softmax(logits_post, dim=-1)
    prior_logprobs = torch.log_softmax(logits_prior, dim=-1)
    if balance == 0.5:
        kl = F.kl_div(prior_logprobs, post_logprobs, reduction='none', log_target=True)
    else:
        kl_a = F.kl_div(prior_logprobs, post_logprobs.detach(), reduction='none', log_target=True)
        kl_b = F.kl_div(prior_logprobs.detach(), post_logprobs, reduction='none', log_target=True)
        kl = (1 - balance) * kl_a + balance * kl_b
    kl = kl.view(orig_shape).sum(-1)
    if reduce == 'sum':
        kl = kl.sum()
    elif reduce == 'mean':
        kl = kl.mean()
    return kl


# classes
class ChamferLossKL(nn.Module):
    """
    Calculates the KL-divergence between two sets of (R.V.) particle coordinates.
    """

    def __init__(self, use_reverse_kl=False):
        super(ChamferLossKL, self).__init__()
        self.use_reverse_kl = use_reverse_kl

    def forward(self, mu_preds, logvar_preds, mu_gts, logvar_gts, posterior_mask=None):
        """
        mu_preds, logvar_preds: [bs, n_x, feat_dim]
        mu_gts, logvar_gts: [bs, n_y, feat_dim]
        posterior_mask: [bs, n_x]
        """
        p_kl = batch_pairwise_kl(mu_preds, logvar_preds, mu_gts, logvar_gts, reverse_kl=False)
        # [bs, n_x, n_y]
        if self.use_reverse_kl:
            p_rkl = batch_pairwise_kl(mu_preds, logvar_preds, mu_gts, logvar_gts, reverse_kl=True)
            p_kl = 0.5 * (p_kl + p_rkl.transpose(2, 1))
        mins, _ = torch.min(p_kl, 1)  # [bs, n_y]
        loss_1 = torch.sum(mins, 1)
        mins, _ = torch.min(p_kl, 2)  # [bs, n_x]
        if posterior_mask is not None:
            mins = mins * posterior_mask
        loss_2 = torch.sum(mins, 1)
        return loss_1 + loss_2


class NetVGGFeatures(nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vggnet = models.vgg16(pretrained=True)
        self.vggnet.eval()
        self.vggnet.requires_grad_(False)
        self.layer_ids = layer_ids

    def forward(self, x):
        output = []
        for i in range(self.layer_ids[-1] + 1):
            x = self.vggnet.features[i](x)

            if i in self.layer_ids:
                output.append(x)

        return output


class VGGDistance(nn.Module):

    def __init__(self, layer_ids=(2, 7, 12, 21, 30), accumulate_mode='sum', device=torch.device("cpu"),
                 normalize=True, use_loss_scale=False, vgg_coeff=0.12151):
        super().__init__()

        self.vgg = NetVGGFeatures(layer_ids).to(device)
        self.layer_ids = layer_ids
        self.accumulate_mode = accumulate_mode
        self.device = device
        self.use_normalization = normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.use_loss_scale = use_loss_scale
        self.vgg_coeff = vgg_coeff

    def forward(self, I1, I2, reduction='sum', only_image=False):
        b_sz = I1.size(0)
        num_ch = I1.size(1)

        if self.accumulate_mode == 'sum':
            loss = ((I1 - I2) ** 2).view(b_sz, -1).sum(1)
            # if normalized, effectively: (1 / (std ** 2)) * (I_1 - I_2) ** 2
        elif self.accumulate_mode == 'ch_mean':
            loss = ((I1 - I2) ** 2).view(b_sz, I1.shape[1], -1).mean(1).sum(-1)
        else:
            loss = ((I1 - I2) ** 2).view(b_sz, -1).mean(1)

        if self.use_normalization:
            I1, I2 = self.normalize(I1), self.normalize(I2)

        if num_ch == 1:
            I1 = I1.repeat(1, 3, 1, 1)
            I2 = I2.repeat(1, 3, 1, 1)

        f1 = self.vgg(I1)
        f2 = self.vgg(I2)

        if not only_image:
            for i in range(len(self.layer_ids)):
                if self.accumulate_mode == 'sum':
                    layer_loss = ((f1[i] - f2[i]) ** 2).view(b_sz, -1).sum(1)
                elif self.accumulate_mode == 'ch_mean':
                    layer_loss = ((f1[i] - f2[i]) ** 2).view(b_sz, f1[i].shape[1], -1).mean(1).sum(-1)
                else:
                    layer_loss = ((f1[i] - f2[i]) ** 2).view(b_sz, -1).mean(1)
                c = self.vgg_coeff if self.use_normalization else 1.0
                loss = loss + c * layer_loss

        if self.use_loss_scale:
            # by using `sum` for the features, and using scaling instead of `mean` we maintain the weight
            # of each dimension contribution to the loss
            max_dim = max([np.product(f.shape[1:]) for f in f1])
            scale = 1 / max_dim
            loss = scale * loss
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def get_dimensions(self, device=torch.device("cpu")):
        dims = []
        dummy_input = torch.zeros(1, 3, 128, 128).to(device)
        dims.append(dummy_input.view(1, -1).size(1))
        f = self.vgg(dummy_input)
        for i in range(len(self.layer_ids)):
            dims.append(f[i].view(1, -1).size(1))
        return dims


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        # self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins, 1)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins, 1)
        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x, device=x.device, dtype=torch.long)
        diag_ind_y = torch.arange(0, num_points_y, device=y.device, dtype=torch.long)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(
            zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P


"""
LPIPS
based on: https://github.com/CompVis/taming-transformers/blob/master/taming
"""


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "eval/lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class LossLPIPS(nn.Module):
    def __init__(self, pixelloss_weight=1.0, perceptual_weight=0.1, normalized_rgb=True):
        super().__init__()
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.normalized_rgb = normalized_rgb

    def scale_input(self, x):
        if self.normalized_rgb:
            return x
        else:
            return 2 * x - 1

    def forward(self, inputs, reconstructions, reduction='mean', split="train", p_loss=True):
        # rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_loss = (inputs.contiguous() - reconstructions.contiguous()) ** 2
        if p_loss and self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(self.scale_input(inputs.contiguous()),
                                          self.scale_input(reconstructions.contiguous()))
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        if reduction == 'mean':
            loss = torch.mean(nll_loss)
        elif reduction == 'sum':
            loss = torch.sum(nll_loss)
        else:
            loss = nll_loss.view(inputs.shape[0], -1).mean(-1, keepdim=True)

        # log = {"total_loss".format(split): loss.clone().detach().mean(),
        #        "nll_loss".format(split): nll_loss.detach().mean(),
        #        "rec_loss".format(split): rec_loss.detach().mean(),
        #        "p_loss".format(split): p_loss.detach().mean(),
        #        }
        # return loss, log
        return loss


if __name__ == '__main__':
    bs = 32
    n_points_x = 10
    n_points_y = 15
    dim = 8
    x = torch.randn(bs, n_points_x, dim)
    y = torch.randn(bs, n_points_y, dim)
    for metric in ['cosine', 'l1', 'l2', 'l2_simple']:
        P = batch_pairwise_dist(x, y, metric)
        print(f'metric: {metric}, P: {P.shape}, max: {P.max()}, min: {P.min()}')



# POINT CLOUD LOSSES


# ---------- curvature weights on x_pts (input) ----------
def estimate_curvature(x, k=16):
    # x: [B,N,3]; return kappa in [B,N] (bigger => more curved/edgey)
    with torch.no_grad():
        # quick kNN via pairwise distances (ok up to medium N)
        d2 = torch.cdist(x, x, p=2) ** 2             # [B,N,N]
        knn_idx = d2.topk(k=k+1, largest=False).indices[..., 1:]  # drop self
        b = torch.arange(x.size(0), device=x.device)[:, None, None]
        nbrs = x[b, knn_idx]                          # [B,N,k,3]
        mu = nbrs.mean(dim=2, keepdim=True)           # [B,N,1,3]
        C = (nbrs - mu).transpose(-1, -2) @ (nbrs - mu) / max(k-1,1)  # [B,N,3,3]
        # curvature ~ smallest eigenvalue / trace; use symmetric eig
        # torch.linalg.eigvals on 3x3 is fine
        evals = torch.linalg.eigvalsh(C).clamp_min(0.)  # [B,N,3]
        kappa = (evals[..., 0] / (evals.sum(dim=-1) + 1e-8))        # [B,N]
        # normalize across N for stability
        kappa = kappa / (kappa.mean(dim=1, keepdim=True) + 1e-6)
        return kappa.clamp(min=0.1, max=5.0)          # avoid extremes

def chamfer_l2_weighted(p, q, q_mask=None, q_weights=None):
    # p: [B,M,3], q: [B,N,3], q_weights: [B,N] positive weights
    B, M, _ = p.shape; N = q.shape[1]
    q_pad = q
    if q_mask is not None:
        big = torch.tensor(1e6, device=q.device, dtype=q.dtype)
        q_pad = torch.where(q_mask[..., None], q, big)
    d_pq = torch.cdist(p, q_pad, p=2) ** 2                       # [B,M,N]
    term_pq = d_pq.min(dim=-1).values.mean(dim=-1)               # [B]

    d_qp = (torch.cdist(q, p, p=2) ** 2).min(dim=-1).values      # [B,N]
    if q_weights is None:
        qw = torch.ones_like(d_qp)
    else:
        qw = q_weights
    if q_mask is not None:
        qw = qw * q_mask.float()
    qw = qw / (qw.sum(dim=-1, keepdim=True).clamp_min(1.0))      # normalize per batch
    term_qp = (d_qp * qw).sum(dim=-1)                            # [B]
    return 0.5 * (term_pq + term_qp)

import torch
import torch.nn.functional as F

def _nn_assign(a, b):
    """
    a: [B,Na,3], b: [B,Nb,3]
    Returns:
      idx: [B,Na]   nearest neighbor index in b for each a_i
      d   : [B,Na]  L2 distance (not squared)
    """
    # pairwise squared distances
    a2 = (a*a).sum(-1, keepdim=True)                   # [B,Na,1]
    b2 = (b*b).sum(-1).unsqueeze(1)                    # [B,1,Nb]
    ab = a @ b.transpose(1, 2)                         # [B,Na,Nb]
    d2 = (a2 + b2 - 2*ab).clamp_min(0.0)
    idx = d2.argmin(dim=2)                             # [B,Na]
    d = d2.gather(2, idx.unsqueeze(-1)).sqrt().squeeze(-1)  # [B,Na]
    return idx, d

def _query_frequency(idx, Nb):
    """
    idx: [B,Na] (indices in [0..Nb-1])
    Returns:
      q: [B,Nb] counts per target point
    """
    B, Na = idx.shape
    q = torch.zeros(B, Nb, device=idx.device, dtype=torch.float32)
    q.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float32))
    # avoid zero for points never queried (kept as 0; we won't index them then)
    return q.clamp_min(1.0)

import torch
import torch.nn.functional as F

@torch.no_grad()
def _batch_tau_from_medians(pred, gt):
    # robust per-batch scale from NN dists (both directions), floored
    # pred:[B,Np,3], gt:[B,Ng,3]
    d_pg = torch.cdist(pred, gt, p=2).min(dim=2).values   # [B,Np]
    d_gp = torch.cdist(gt, pred, p=2).min(dim=2).values   # [B,Ng]
    med = 0.5 * (d_pg.median(dim=1).values + d_gp.median(dim=1).values)  # [B]
    med = torch.nan_to_num(med, nan=1e-2, posinf=1.0, neginf=1e-4)
    return med.clamp_min(1e-3).unsqueeze(-1)  # [B,1]

def density_aware_chamfer(pred, gt, tau=None, freq_temp=1.0, eps=1e-6):
    """
    Stable DCD (Wu et al., NeurIPS'21) with clamps/nan guards.
    pred, gt: [B,N,3] (already downsampled)
    tau: None -> per-batch auto; else float or [B,1]
    """
    # sanitize inputs
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e3, neginf=-1e3).clamp_(-2.0, 2.0)
    gt   = torch.nan_to_num(gt,   nan=0.0, posinf=1e3, neginf=-1e3).clamp_(-2.0, 2.0)

    B, Np, _ = pred.shape
    Ng       = gt.shape[1]

    # NN assignments (both directions)
    d_pg = torch.cdist(pred, gt, p=2)                    # [B,Np,Ng]
    d_gp = torch.cdist(gt, pred, p=2)                    # [B,Ng,Np]
    dp  = d_pg.min(dim=2).values                         # [B,Np]
    dg  = d_gp.min(dim=2).values                         # [B,Ng]
    idx_pg = d_pg.argmin(dim=2)                          # [B,Np]
    idx_gp = d_gp.argmin(dim=2)                          # [B,Ng]

    # query frequency
    qg = torch.zeros(B, Ng, device=gt.device, dtype=gt.dtype)
    qp = torch.zeros(B, Np, device=pred.device, dtype=pred.dtype)
    qg.scatter_add_(1, idx_pg, torch.ones_like(idx_pg, dtype=gt.dtype))
    qp.scatter_add_(1, idx_gp, torch.ones_like(idx_gp, dtype=pred.dtype))
    w_pg = 1.0 / (torch.pow(qg.gather(1, idx_pg), freq_temp) + eps)  # [B,Np]
    w_gp = 1.0 / (torch.pow(qp.gather(1, idx_gp), freq_temp) + eps)  # [B,Ng]

    # tau
    if tau is None:
        tau_val = _batch_tau_from_medians(pred, gt)                 # [B,1]
    else:
        tau_val = torch.as_tensor(tau, device=pred.device, dtype=pred.dtype).view(1, 1).clamp_min(1e-3)

    # bounded cost
    c_pg = 1.0 - torch.exp(-dp / (tau_val + eps))                   # [B,Np]
    c_gp = 1.0 - torch.exp(-dg / (tau_val + eps))                   # [B,Ng]

    term_pg = (w_pg * c_pg).mean(dim=1)                             # [B]
    term_gp = (w_gp * c_gp).mean(dim=1)                             # [B]
    return 0.5 * (term_pg + term_gp)                                # [B]

        # ---------- weighted DCD ----------
def weighted_dcd(pred, gt, w_pred=None, tau=None, freq_temp=1.0, eps=1e-8):
        """
        pred:  [B, M_p, 3]
        gt:    [B, N, 3]
        w_pred:[B, M_p, 1] in [0,1] (weights for pred points). If None -> unweighted DCD.
        Returns per-batch scalar [B].
        """
        if w_pred is None:
            return density_aware_chamfer(pred=pred, gt=gt, tau=tau, freq_temp=freq_temp)

        # distances
        d = torch.cdist(pred, gt, p=2)  # [B, M_p, N]

        # pred -> gt : min over gt, then **weighted** mean over pred
        dmin_p = d.min(dim=-1).values                  # [B, M_p]
        term_p = (dmin_p * w_pred.squeeze(-1)).sum(dim=1)  # [B]

        # gt -> pred : we approximate a weight-aware "soft-min over pred"
        # Use pred weights as an attention over pred points.
        att = (w_pred.squeeze(-1) / (w_pred.squeeze(-1).sum(dim=1, keepdim=True).clamp_min(eps)))  # [B,M_p]
        dmin_q = (att.unsqueeze(-1) * d).sum(dim=1)   # [B, N]
        term_q = dmin_q.mean(dim=1)                   # [B] plain mean over gt

        return 0.5 * (term_p + term_q)

@torch.no_grad()
def _tile_ranges(n, tile):
    i = 0
    while i < n:
        j = min(i + tile, n)
        yield i, j
        i = j
def repulsion_loss_weighted_knn(P, w=None, k=6, r=0.03, tile=4096, stopgrad_w=True):
    B, M, _ = P.shape
    if w is None:
        w = P.new_ones(B, M, 1)
    if stopgrad_w:
        w = w.detach()

    total = P.new_zeros(())
    count = 0
    BIG = 1e6

    for b in range(B):
        X = P[b]                       # [M,3]
        W = w[b].clamp(0, 1)          # [M,1]

        keep = (W.squeeze(-1) > 1e-3)
        X = X[keep]
        W = W[keep]
        Me = X.shape[0]
        if Me <= 1:
            continue

        acc = X.new_zeros(Me, k)      # store k nn distances

        for i0 in range(0, Me, tile):
            i1 = min(i0 + tile, Me)
            Q = X[i0:i1]                              # [Ti,3]
            D = torch.cdist(Q, X, p=2)               # [Ti, Me]

            # DO NOT modify D in place. Create a new tensor with big values at (near-)self entries.
            # self pairs show up as ~0 distance; push them out before topk:
            D_masked = torch.where(D <= 1e-12, D.new_full(D.shape, BIG), D)

            vals, _ = torch.topk(D_masked, k=k, largest=False, dim=-1)  # [Ti,k]
            acc[i0:i1] = vals

        rep = torch.relu(r - acc)         # [Me,k]
        rep = rep * W                     # weight each point
        total = total + rep.mean()
        count += 1

    return total / max(count, 1)


# utils/loss_functions.py
def coverage_loss_weighted(P_pred, W_pred, P_gt, max_pred=12000, max_gt=12000, stopgrad_w=True):
    """
    P_pred : [B, M, 3]
    W_pred : [B, M, 1] (soft top-k point weights) — will downweight inactive points
    P_gt   : [B, N, 3]
    """
    B, M, _ = P_pred.shape
    if stopgrad_w:
        W_pred = W_pred.detach()
    out = P_pred.new_zeros(B)

    for b in range(B):
        X = P_pred[b]
        W = W_pred[b].clamp(0,1)
        Y = P_gt[b]

        # sample proportional to weights to focus on effective points
        m = X.shape[0]
        n = Y.shape[0]

        if m > max_pred:
            probs = (W.squeeze(-1) + 1e-6)
            idx = torch.multinomial(probs, num_samples=max_pred, replacement=False)
            X = X[idx]; W = W[idx]

        if n > max_gt:
            # uniform subsample gt is fine
            jdx = torch.randperm(n, device=Y.device)[:max_gt]
            Y = Y[jdx]

        D = torch.cdist(X, Y, p=2)       # [Mp, Ng]
        dmin, _ = D.min(dim=1)           # [Mp]
        # weight by W and average
        out[b] = (dmin * W.squeeze(-1)).sum() / (W.sum().clamp_min(1.0))

    return out.mean()


def repulsion_loss(pts, h=0.02):
    """
    Light repulsion to prevent clumping (optional).
    pts: [B,N,3]
    """
    B, N, _ = pts.shape
    if N < 2: return pts.sum()*0.0
    # quick block-wise approximation: sample K neighbors via random shuffle
    idx = torch.randperm(N, device=pts.device)
    nbr = pts[:, idx, :]
    d2 = ((pts - nbr)**2).sum(-1)
    return F.relu(h*h - d2).mean()

def calc_pc_dcd_loss(model_output,
                     pts_gt,                 # [B, N, 3] (you said you’ll pre-downsample to N)
                     *,
                     use_color=False,
                     rgb_gt=None,            # [B, N, 3] if use_color
                     w_color=0.0,
                     w_repulsion=0.05,
                     tau=None,
                     freq_temp=1.0,
                     kl_static=None, w_kl=0.0,
                     loss_obj_reg=None, w_obj=0.0):
    """
    Returns: total_loss, dict_of_terms
    Expects in model_output:
      - 'points_scene': [B, N, 3]
      - optional 'rgb_scene': [B, N, 3]
    """
    pts_pred = model_output["points_scene"]            # [B, N, 3]

    # DCD recon
    loss_rec = density_aware_chamfer(pts_pred, pts_gt, tau=tau, freq_temp=freq_temp)

    # Optional: color via NN from pred->gt (uses same NN as inside DCD semantics)
    loss_color = pts_pred.sum()*0.0
    if use_color and (rgb_gt is not None) and ("rgb_scene" in model_output):
        # reuse NN by recomputing pred->gt indices once (cheap)
        idx_pg, _ = _nn_assign(pts_pred, pts_gt)      # [B, N]
        rgb_pred = model_output["rgb_scene"]          # [B, N, 3]
        rgb_match = pts_gt.new_empty(rgb_pred.shape)  # [B, N, 3]
        rgb_match.scatter_(1, torch.arange(rgb_pred.size(1), device=pts_pred.device)[None, :, None].expand_as(rgb_pred),
                           rgb_gt.gather(1, idx_pg.unsqueeze(-1).expand(-1, -1, 3)))
        loss_color = F.l1_loss(rgb_pred, rgb_match)

    # Optional: repulsion (keeps spawned points from collapsing)
    loss_rep = repulsion_loss(pts_pred) if w_repulsion > 0 else pts_pred.sum()*0.0

    # KL / object regs (pass from your forward)
    loss_kl  = kl_static   if (kl_static   is not None) else pts_pred.sum()*0.0
    loss_obj = loss_obj_reg if (loss_obj_reg is not None) else pts_pred.sum()*0.0

    total = loss_rec + w_color*loss_color + w_repulsion*loss_rep + w_kl*loss_kl + w_obj*loss_obj

    logs = {
        "rec_dcd": loss_rec.detach(),
        "color":   loss_color.detach(),
        "repulsion": loss_rep.detach(),
        "KL": (loss_kl.detach() if torch.is_tensor(loss_kl) else torch.tensor(0.0, device=pts_pred.device)),
        "obj_reg": (loss_obj.detach() if torch.is_tensor(loss_obj) else torch.tensor(0.0, device=pts_pred.device)),
        "loss_total": total.detach(),
    }
    return total, logs




def repulsion_loss(P, k=8, r_frac=0.01):
    # P: [B,M,3]; radius r as % of scene diagonal (assuming coords in [-1,1])
    with torch.no_grad():
        scene_diag = 2.0  # [-1,1] cube -> diagonal ~2*sqrt(3); using 2.0 as scale is fine
        r = r_frac * scene_diag
    d2 = torch.cdist(P, P, p=2) ** 2                            # [B,M,M]
    knn = d2.topk(k=k+1, largest=False).values[..., 1:]         # [B,M,k]
    d = knn.sqrt()                                              # [B,M,k]
    return torch.relu(r - d).mean()


def coverage_loss(p, q, q_mask=None, tau=0.01):
    # distance from each q to nearest p, hinge at tau
    d = (torch.cdist(q, p, p=2)).min(dim=-1).values  # [B,N]
    if q_mask is not None:
        d = d * q_mask.float()
        denom = q_mask.sum(dim=-1).clamp_min(1).float()
        return torch.relu(d - tau).sum(dim=-1) / denom
    return torch.relu(d - tau).mean(dim=-1)


def estimate_normals_pca(P, k=16):
    # P: [B,M,3] -> normals [B,M,3] (unit)
    d2 = torch.cdist(P, P, p=2) ** 2
    knn_idx = d2.topk(k=k+1, largest=False).indices[..., 1:]       # [B,M,k]
    b = torch.arange(P.size(0), device=P.device)[:, None, None]
    nbrs = P[b, knn_idx]                                           # [B,M,k,3]
    mu = nbrs.mean(dim=2, keepdim=True)
    C = (nbrs - mu).transpose(-1, -2) @ (nbrs - mu) / max(k-1,1)   # [B,M,3,3]
    evals, evecs = torch.linalg.eigh(C)                            # ascending
    n = evecs[..., 0]                                              # smallest eigenvector
    return torch.nn.functional.normalize(n, dim=-1)




def kp_separation_loss(kp_xyz, obj_on, delta=0.05):
    # kp_xyz: [B,K,3], obj_on: [B,K,1] in [0,1]
    w = obj_on.squeeze(-1)                              # [B,K]
    # pairwise distances
    diff = kp_xyz.unsqueeze(2) - kp_xyz.unsqueeze(1)    # [B,K,K,3]
    d = diff.norm(dim=-1) + torch.eye(kp_xyz.size(1), device=kp_xyz.device)[None]*1e9
    # weight by presence of both kps
    w_pair = (w.unsqueeze(2) * w.unsqueeze(1))          # [B,K,K]
    sep = torch.relu(delta - d) * w_pair
    return sep.mean()

