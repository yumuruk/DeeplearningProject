import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class UIEBDataset(Dataset):
    def __init__(self, root, transform=None, img_size = 256, is_test = False):
        if transform is not None:
            self.transform = T.Compose(transform)
        else:
            self.transform = T.Compose(
                [T.Resize((img_size, img_size)),
                 T.ToTensor(),]
            )
        self.input_files, self.gt_files, self.t_p_files, self.B_p_files = self.get_file_paths(root, is_test)
        self.len = min(len(self.input_files), len(self.gt_files), len(self.t_p_files), len(self.B_p_files))
        
    def __getitem__(self, index):
        input_image = Image.open(self.input_files[index % self.len])
        gt_image = Image.open(self.gt_files[index % self.len])
        t_p = Image.open(self.t_p_files[index % self.len])
        B_p = Image.open(self.B_p_files[index % self.len])
        input_image = self.transform(input_image)
        gt_image = self.transform(gt_image)
        t_p = self.transform(t_p)
        B_p = self.transform(B_p)
        return {"inp": input_image, "gt": gt_image, "t": t_p, "B": B_p}
    
    def __len__(self):
        return self.len
    
    def get_file_paths(self, root, is_test):
        if is_test == True:
            input_files = sorted(glob.glob(os.path.join(root, 'images') + "/*.*"))
            t_p_files = sorted(glob.glob(os.path.join(root, 't_prior_7') + "/*.*"))
            B_p_files = sorted(glob.glob(os.path.join(root, 'B_prior_7') + "/*.*"))
            gt_files = []
        else:
            input_files = sorted(glob.glob(os.path.join(root, 'images') + "/*.*"))
            gt_files = sorted(glob.glob(os.path.join(root, 'gt') + "/*.*"))
            t_p_files = sorted(glob.glob(os.path.join(root, 't_prior_7') + "/*.*"))
            B_p_files = sorted(glob.glob(os.path.join(root, 'B_prior_7') + "/*.*"))
        return input_files, gt_files, t_p_files, B_p_files
    