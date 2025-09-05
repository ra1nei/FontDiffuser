import torch
import torch.nn as nn
import src.modules.scr_modules as SCRModules

from info_nce import InfoNCE
import kornia.augmentation as K

class SCR(nn.Module):
    def __init__(self, 
                 temperature, 
                 mode='training',
                 image_size=96,
                 loss_mode='intra',
                 alpha_intra=0.3,
                 beta_cross=0.7):
        """
        loss_mode:
            - 'intra': chỉ dùng intra-positive/negative (gốc)
            - 'cross': chỉ dùng cross-positive/negative
            - 'both': dùng cả intra và cross
        """
        super().__init__()
        style_vgg = SCRModules.vgg
        style_vgg = nn.Sequential(*list(style_vgg.children()))
        self.StyleFeatExtractor = SCRModules.StyleExtractor(encoder=style_vgg)  
        self.StyleFeatProjector = SCRModules.Projector()

        if mode == 'training':
            self.StyleFeatExtractor.requires_grad_(True)
            self.StyleFeatProjector.requires_grad_(True)
        else:
            self.StyleFeatExtractor.requires_grad_(False)
            self.StyleFeatProjector.requires_grad_(False)
        
        # NCE Loss
        self.nce_loss = InfoNCE(temperature=temperature, negative_mode='paired')

        # Pos Image random resize and crop
        self.patch_sampler = K.RandomResizedCrop(
            (image_size, image_size),
            scale=(0.8, 1.0),
            ratio=(0.75, 1.33)
        )

        # Chọn chế độ loss
        assert loss_mode in ['intra', 'cross', 'both']
        self.loss_mode = loss_mode
        self.alpha_intra = alpha_intra
        self.beta_cross = beta_cross

    def _extract_embeddings(self, imgs, nce_layers):
        """Helper: lấy style embedding cho 1 batch ảnh"""
        return self.StyleFeatProjector(
            self.StyleFeatExtractor(imgs, nce_layers),
            nce_layers
        )

    def forward(self,
                sample_imgs,
                intra_pos_imgs=None,
                cross_pos_imgs=None,
                intra_neg_imgs=None,
                cross_neg_imgs=None,
                nce_layers='0,1,2,3,4,5'):
        """
        Forward trả ra embedding theo mode đã chọn
        """
        # Anchor
        sample_style_embeddings = self._extract_embeddings(sample_imgs, nce_layers)

        # Intra-positive
        if self.loss_mode in ['intra', 'both'] and intra_pos_imgs is not None:
            intra_pos_imgs = self.patch_sampler(intra_pos_imgs)
            intra_pos_style_embeddings = self._extract_embeddings(intra_pos_imgs, nce_layers)
        else:
            intra_pos_style_embeddings = None

        # Cross-positive
        if self.loss_mode in ['cross', 'both'] and cross_pos_imgs is not None:
            cross_pos_imgs = self.patch_sampler(cross_pos_imgs)
            cross_pos_style_embeddings = self._extract_embeddings(cross_pos_imgs, nce_layers)
        else:
            cross_pos_style_embeddings = None

        # Intra-negative
        intra_neg_embeddings = None
        if self.loss_mode in ['intra', 'both'] and intra_neg_imgs is not None:
            _, num_neg, _, _, _ = intra_neg_imgs.shape
            for i in range(num_neg):
                neg_once = intra_neg_imgs[:, i, :, :]
                neg_embed_once = self._extract_embeddings(neg_once, nce_layers)
                for j, layer_out in enumerate(neg_embed_once):
                    if j == 0:
                        neg_mid = layer_out[None, :, :]
                    else:
                        neg_mid = torch.cat([neg_mid, layer_out[None, :, :]], dim=0)
                if i == 0:
                    intra_neg_embeddings = neg_mid[:, :, None, :]
                else:
                    intra_neg_embeddings = torch.cat([intra_neg_embeddings, neg_mid[:, :, None, :]], dim=2)

        # Cross-negative
        cross_neg_embeddings = None
        if self.loss_mode in ['cross', 'both'] and cross_neg_imgs is not None:
            _, num_neg, _, _, _ = cross_neg_imgs.shape
            for i in range(num_neg):
                neg_once = cross_neg_imgs[:, i, :, :]
                neg_embed_once = self._extract_embeddings(neg_once, nce_layers)
                for j, layer_out in enumerate(neg_embed_once):
                    if j == 0:
                        neg_mid = layer_out[None, :, :]
                    else:
                        neg_mid = torch.cat([neg_mid, layer_out[None, :, :]], dim=0)
                if i == 0:
                    cross_neg_embeddings = neg_mid[:, :, None, :]
                else:
                    cross_neg_embeddings = torch.cat([cross_neg_embeddings, neg_mid[:, :, None, :]], dim=2)

        return (sample_style_embeddings,
                intra_pos_style_embeddings,
                cross_pos_style_embeddings,
                intra_neg_embeddings,
                cross_neg_embeddings)

    # TODO
    def calculate_nce_loss(self,
                           sample_s,
                           intra_pos_s=None,
                           cross_pos_s=None,
                           intra_neg_s=None,
                           cross_neg_s=None):
        """
        Tính loss theo loss_mode
        """
        total_loss = 0.
        count = 0

        # Intra
        if self.loss_mode in ['intra', 'both'] and intra_pos_s is not None and intra_neg_s is not None:
            num_layer = intra_neg_s.shape[0]
            intra_loss = 0.
            for layer, (sample, pos, neg) in enumerate(zip(sample_s, intra_pos_s, intra_neg_s)):
                intra_loss += self.nce_loss(sample, pos, neg)
            intra_loss = intra_loss / num_layer
            if self.loss_mode == 'both':
                total_loss += self.alpha_intra * intra_loss
            else:
                total_loss += intra_loss
            count += 1

        # Cross
        if self.loss_mode in ['cross', 'both'] and cross_pos_s is not None and cross_neg_s is not None:
            num_layer = cross_neg_s.shape[0]
            cross_loss = 0.
            for layer, (sample, pos, neg) in enumerate(zip(sample_s, cross_pos_s, cross_neg_s)):
                cross_loss += self.nce_loss(sample, pos, neg)
            cross_loss = cross_loss / num_layer
            if self.loss_mode == 'both':
                total_loss += self.beta_cross * cross_loss
            else:
                total_loss += cross_loss
            count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0, device=sample_s[0].device)
