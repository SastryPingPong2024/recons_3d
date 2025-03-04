import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2
from models import modify_resnet
import random
from PIL import Image
import torchvision.transforms as T

transform = T.Compose([
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.25),  # Adjust brightness, contrast, etc.
    # T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Apply Gaussian blur
    T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),  # Sharpening effect
    # T.RandomGrayscale(p=0.2),  # Convert to grayscale with probability 20%
    T.ToTensor(),  # Convert to tensor
])

class TableTennisDataset(Dataset):
    def __init__(self, data_root, transform=None, augment=True):
        """
        Args:
            data_root (string): Root directory of the dataset.
                                Should contain folders like match1, match2, etc.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.data_root = data_root
        if augment:
            suffix = '_aug'
        else: 
            suffix = ''
        self.images_root = os.path.join(data_root, 'images'+suffix)
        self.labels_root = os.path.join(data_root, 'labels'+suffix)
        self.transform = transform
        self.samples = []

        # Collect all frame and label paths
        for item in os.listdir(self.images_root):
            image_path = os.path.join(self.images_root, item)
            label_path = os.path.join(self.labels_root, item.replace('.jpg', '.csv'))
            self.samples.append({
                'image_path': image_path,
                'label_path': label_path,
            })
        
        # Total number of frames
        self.total_frames = len(self.samples)
    
    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, idx):
        # Get sample information
        sample_info = self.samples[idx]
        frame_path = sample_info['image_path']
        label_path = sample_info['label_path']
        # print(frame_path, label_path)
        # Load frame
        frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        w = frame.shape[1]
        h = frame.shape[0]
        
        # Load labels
        label = np.loadtxt(label_path)

        l = int(np.min(label[:,0]))
        r = int(np.max(label[:,0]))
        t = int(np.min(label[:,1]))
        d = int(np.max(label[:,1]))
        
        v1 = np.random.randint(5, 50)
        t -= v1
        if t < 0:
            v1 += t
            t = 0
        
        v2 = np.random.randint(5, 50)
        d += v2
        if d > h:
            v2 -= (d-h)
            d = h
        # print(v1,v2)
        l -= np.random.randint(5, 50)
        if l < 0:
            l = 0
        
        r += np.random.randint(5, 50)
        if r > w:
            r = w
        
        l_orig = l
        r_orig = r
        t_orig = t
        d_orig = d
        
        if (d-t) > (r-l):
            r += int(((d-t) - (r-l))/2)
            l = r - (d-t)
            if l < 0:
                l = 0
                r = d-t
            if r > w:
                r = w
                l = w - (d-t)
        else:
            d += int(((r-l) - (d-t))/2)
            t = d - (r-l)
            if t < 0:
                t = 0
                d = r-l
            if d > h:
                d = h
                t = h - (r-l)
        label[:,0] = (label[:,0] - l) / (r-l)
        label[:,1] = (label[:,1] - t) / (d-t)

        image = frame[t:d, l:r,:]

        image[:int(t_orig-t),:,:] = 0
        image[-int(d-d_orig+1):,:,:] = 0
        image[:,:int(l_orig-l),:] = 0
        image[:,-int(r-r_orig+1):,:] = 0
        
        # reshape image to (224, 224)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Convert from BGR to RGB color space (OpenCV uses BGR)
        cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(cv2_image.shape)
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2_image) # Shape: (H, W, C)
        
        image_transformed = transform(pil_image)
        # print(image_transformed.shape)
        # transform image to (3,224,224)
        # image = image.transpose((2, 0, 1))

        # # normalize image and convert to float
        # image = image / 255.0
        # image = image.astype(np.float32)
        # print(np.min(image),np.max(image))

        # flatten label and convert to float
        label = label.flatten()
        label = label.astype(np.float32)

        # Convert tensor back to PIL for applying transforms
        # image_pil = T.ToPILImage()(image)

        # Apply transforms
        
        # print(torch.min(image_transformed))
        # print(torch.max(image_transformed))
        # Convert back to tensor
        # image_transformed = T.ToTensor()(image_transformed)  # Shape: (3, H, W)
        # image = transform(image)
        return image_transformed, label
    
if __name__ == "__main__":
    # Define the dataset
    dataset = TableTennisDataset(data_root='dataset/train/')

    # Get a sample
    image, label = dataset[0]

    model = modify_resnet(output_size=20)
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()
    pred_label = model(torch.tensor(image).unsqueeze(0)).detach().cpu().numpy()
    print(np.mean((pred_label-label)**2))
    # Convert image to cv2 format
    image = (np.array(image) * 255).astype(np.uint8).transpose((1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    label = label.reshape(-1, 2)
    pred_label = pred_label.reshape(-1, 2)
    image_marked = image.copy()
    # print(label, pred_label)
    i = 0
    for l in label:
        i += 1
        x = int(l[0] * 224)
        y = int(l[1] * 224)
        cv2.circle(image_marked, (x, y), 3, (0, 25*i, 0), -1)
    for l in pred_label:
        x = int(l[0] * 224)
        y = int(l[1] * 224)
        cv2.circle(image_marked, (x, y), 3, (255, 0, 0), -1)

    # Display the image and label
    cv2.imshow("image", image_marked)
    
    # wait for a key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(label)
    