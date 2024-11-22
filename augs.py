
from torchvision import transforms


class ViewAugmentations:
    
    def __init__(self):
        self.augmentations = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.4, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])

        self.normalize_tensorize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
        
    def apply_augmentation(self, x):
        # x: PIL Image
        # return: 3, H, W
        return self.augmentations(x)
    
    def normalize_tensorize(self, x):
        # x: PIL image 
        # reutrn: 3, H, W
        return self.normalize_tensorize(x)
    