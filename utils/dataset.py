# import os
# import torch
# import numpy as np
# from PIL import Image
# from torch.utils.data import Dataset
# import torchvision.transforms.functional as TF
# import torchvision.transforms as transforms
# import random

# # ============ ENHANCED AUGMENTATION FOR PADDY FIELD WITH NOISY BACKGROUNDS ============
# class PaddyAugmentation(object):
#     """
#     Advanced augmentation strategy for paddy field leaf disease segmentation.
#     Handles multiple leaves and noisy backgrounds.
#     """
#     def __init__(self, image_size=(640, 480), apply_probability=0.8):
#         self.image_size = image_size
#         self.apply_probability = apply_probability
        
#     def __call__(self, image, mask):
#         """Apply augmentation to both image and mask consistently"""
        
#         # 1. Random rotation (-15 to 15 degrees) - leaves can be at various angles
#         if random.random() < self.apply_probability:
#             angle = random.uniform(-15, 15)
#             image = TF.rotate(image, angle, fill=0)
#             mask = TF.rotate(mask, angle, fill=0)
        
#         # 2. Random horizontal flip (common for crop images)
#         if random.random() < 0.5:
#             image = TF.hflip(image)
#             mask = TF.hflip(mask)
        
#         # 3. Random vertical flip
#         if random.random() < 0.5:
#             image = TF.vflip(image)
#             mask = TF.vflip(mask)
        
#         # 4. Random affine transform (translate + scale) for multiple leaf positions
#         if random.random() < self.apply_probability:
#             params = transforms.RandomAffine.get_params(
#                 degrees=[0, 0],
#                 translate=[0.1, 0.1],  # ±10% translation
#                 scale_ranges=[0.9, 1.1],  # ±10% scale
#                 shears=[0, 0],
#                 img_size=self.image_size
#             )
#             image = TF.affine(image, *params, fill=0)
#             mask = TF.affine(mask, *params, fill=0)
        
#         # 5. Random elastic deformation for organic leaf shapes
#         if random.random() < self.apply_probability * 0.5:
#             image, mask = self._elastic_deform(image, mask)
        
#         # 6. Random brightness and contrast for varied lighting conditions
#         if random.random() < self.apply_probability:
#             brightness_factor = random.uniform(0.8, 1.3)
#             image = TF.adjust_brightness(image, brightness_factor)
        
#         # 7. Random contrast adjustment for shadow/highlight variations
#         if random.random() < self.apply_probability:
#             contrast_factor = random.uniform(0.8, 1.3)
#             image = TF.adjust_contrast(image, contrast_factor)
        
#         # 8. Random hue and saturation for crop variability
#         if random.random() < self.apply_probability:
#             saturation_factor = random.uniform(0.8, 1.2)
#             image = TF.adjust_saturation(image, saturation_factor)
        
#         # 9. Gaussian blur to handle focus variations
#         if random.random() < self.apply_probability * 0.3:
#             image = TF.gaussian_blur(image, kernel_size=3)
        
#         # 10. Random crop and resize for robustness to multiple leaves
#         if random.random() < self.apply_probability * 0.4:
#             image, mask = self._random_crop_and_resize(image, mask)
        
#         return image, mask
    
#     def _elastic_deform(self, image, mask, alpha=30, sigma=5):
#         """Apply elastic deformation to handle organic leaf shapes"""
#         try:
#             img_array = np.array(image)
#             mask_array = np.array(mask)
            
#             # Generate random displacement fields
#             random_state = np.random.RandomState(None)
#             h, w = img_array.shape[:2]
            
#             dx = random_state.randn(h, w) * sigma
#             dy = random_state.randn(h, w) * sigma
            
#             x, y = np.meshgrid(np.arange(w), np.arange(h))
#             indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            
#             if len(img_array.shape) == 3:
#                 distorted_img = np.zeros_like(img_array)
#                 for c in range(img_array.shape[2]):
#                     distorted_img[:, :, c] = np.reshape(
#                         np.random.choice(img_array[:, :, c].flatten(), size=h*w),
#                         (h, w)
#                     )
            
#             distorted_mask = np.reshape(
#                 np.random.choice(mask_array.flatten(), size=h*w),
#                 (h, w)
#             )
            
#             return Image.fromarray(distorted_img.astype('uint8')), Image.fromarray(distorted_mask.astype('uint8'))
#         except:
#             return image, mask
    
#     def _random_crop_and_resize(self, image, mask, crop_scale=(0.8, 1.0)):
#         """Random crop for handling multiple leaves at different scales"""
#         w, h = image.size
#         crop_size_w = int(w * random.uniform(*crop_scale))
#         crop_size_h = int(h * random.uniform(*crop_scale))
        
#         i = random.randint(0, h - crop_size_h)
#         j = random.randint(0, w - crop_size_w)
        
#         image = TF.crop(image, i, j, crop_size_h, crop_size_w)
#         mask = TF.crop(mask, i, j, crop_size_h, crop_size_w)
        
#         # Resize back to original size
#         image = image.resize((w, h), Image.BILINEAR)
#         mask = mask.resize((w, h), Image.NEAREST)
        
#         return image, mask


# class PaddyBinaryDataset(Dataset):
#     def __init__(self, images_dir, masks_dir, augment=False):
#         self.images_dir = images_dir
#         self.masks_dir = masks_dir
#         self.augment = augment
#         # ========== OLD VERSION (without augmentation) ==========
#         # valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
#         # self.image_files = [file for file in os.listdir(images_dir) if not file.startswith('.') and file.lower().endswith(valid_exts)]
#         # ========== END OLD VERSION ==========
        
#         # ========== NEW VERSION with augmentation support ==========
#         valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
#         self.image_files = [file for file in os.listdir(images_dir) if not file.startswith('.') and file.lower().endswith(valid_exts)]
#         self.augmentation = PaddyAugmentation(image_size=(640, 480), apply_probability=0.8)
#         # ========== END NEW VERSION ==========

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, i):
#         img_filename = self.image_files[i]
#         idx = os.path.splitext(img_filename)[0]

#         img_file = os.path.join(self.images_dir, img_filename)
        
#         mask_file = os.path.join(self.masks_dir, idx + '.png')
#         if not os.path.exists(mask_file):
#             mask_file = os.path.join(self.masks_dir, img_filename)

#         img = Image.open(img_file).convert('RGB')
#         mask = Image.open(mask_file).convert('L') # Ensure mask is read in grayscale

#         # ========== OLD VERSION (simple resize) ==========
#         # img = img.resize((640, 480), Image.BILINEAR)
#         # mask = mask.resize((640, 480), Image.NEAREST)
#         # ========== END OLD VERSION ==========
        
#         # ========== NEW VERSION (augmentation-aware) ==========
#         # Apply augmentation BEFORE resize if training
#         if self.augment:
#             img, mask = self.augmentation(img, mask)
        
#         # Standardize resolution for efficient batching
#         img = img.resize((640, 480), Image.BILINEAR)
#         mask = mask.resize((640, 480), Image.NEAREST)
#         # ========== END NEW VERSION ==========

#         img_tensor = TF.to_tensor(img)
        
#         # ========== OLD VERSION (simple binary mask) ==========
#         # Enforce strict binary ground truth: black backgrounds stay 0, everything else becomes 1.0
#         # mask_array = np.array(mask)
#         # binary_mask = (mask_array > 0).astype(np.float32)
#         # ========== END OLD VERSION ==========
        
#         # ========== NEW VERSION (with background noise handling) ==========
#         # Handle noisy backgrounds: apply threshold for cleaner binary mask
#         mask_array = np.array(mask)
#         # Use stricter threshold (>127 instead of >0) to reduce noise from gray backgrounds
#         binary_mask = (mask_array > 127).astype(np.float32)
#         # ========== END NEW VERSION ==========
        
#         # Add channel dimension to match model output: (1, H, W)
#         mask_tensor = torch.as_tensor(binary_mask).unsqueeze(0)

#         return {
#             'image': img_tensor,
#             'mask': mask_tensor
#         }

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random

class PaddyBinaryDataset(Dataset):
    def __init__(self, images_dir, masks_dir, is_train=True, patch_size=256):
        """
        is_train: If True, applies random 256x256 cropping and augmentations (speeds up training).
                  If False, resizes to a standardized dimension for validation.
        patch_size: The size of the crop for the training patches.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.is_train = is_train
        self.patch_size = patch_size
        
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        self.image_files = [file for file in os.listdir(images_dir) if not file.startswith('.') and file.lower().endswith(valid_exts)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        img_filename = self.image_files[i]
        idx = os.path.splitext(img_filename)[0]

        img_file = os.path.join(self.images_dir, img_filename)
        
        mask_file = os.path.join(self.masks_dir, idx + '.png')
        if not os.path.exists(mask_file):
            mask_file = os.path.join(self.masks_dir, img_filename)

        img = Image.open(img_file).convert('RGB')
        mask = Image.open(mask_file).convert('L') # Ensure mask is read in grayscale

        # ============ FAST CROP POLICY (From efficientunet--) ============
        if self.is_train:
            # 1. Safety pad: in case any original image is smaller than patch_size
            w, h = img.size
            pad_w = max(0, self.patch_size - w)
            pad_h = max(0, self.patch_size - h)
            if pad_w > 0 or pad_h > 0:
                img = TF.pad(img, (0, 0, pad_w, pad_h))
                mask = TF.pad(mask, (0, 0, pad_w, pad_h))

            # 2. Random Crop (Extracts a smaller patch, making training faster)
            i_crop, j_crop, h_crop, w_crop = transforms.RandomCrop.get_params(img, output_size=(self.patch_size, self.patch_size))
            img = TF.crop(img, i_crop, j_crop, h_crop, w_crop)
            mask = TF.crop(mask, i_crop, j_crop, h_crop, w_crop)
            
            # 3. Geometric Augmentations
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
                
            # 4. Environmental Lighting
            if random.random() > 0.3:
                color_tf = transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3)
                img = color_tf(img)
            
            if random.random() > 0.7:
                img = TF.gaussian_blur(img, kernel_size=[3, 3])
                
        else:
            # Standardize resolution for efficient batching during validation
            img = img.resize((640, 480), Image.BILINEAR)
            mask = mask.resize((640, 480), Image.NEAREST)
        # =================================================================

        img_tensor = TF.to_tensor(img)
        
        # Handle noisy backgrounds: apply threshold for cleaner binary mask
        mask_array = np.array(mask)
        # Use stricter threshold (>127) to reduce noise from gray backgrounds
        binary_mask = (mask_array > 127).astype(np.float32)
        
        # Add channel dimension to match model output: (1, H, W)
        mask_tensor = torch.as_tensor(binary_mask).unsqueeze(0)

        return {
            'image': img_tensor,
            'mask': mask_tensor
        }