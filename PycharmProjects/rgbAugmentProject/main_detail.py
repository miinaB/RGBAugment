import cv2
import torch
import os
import argparse
import matplotlib.pyplot as plt
class Augmentation:
    def __init__(self,path):
        '''
        Load image and convert it to RGB color
        :param path: path to image:
        '''
        self.path=os.path.join("dataset/animal_data", path)
        self.filename=""
        self.pathlist=["./dataset/depth/"+path,"./dataset/thermal/"+path,"./dataset/event/"+path]
        for path in self.pathlist:
            if not os.path.exists(path):
                os.makedirs(path)

    def __getitem__(self):
        '''
        returns origianl image
        :return: Image
        '''
        return self.img
    def print_image(self,img,colormap):
        if colormap=='':
            plt.imshow(img)
            # plt.show()
            return

        plt.imshow(img, cmap=colormap)
        plt.axis('off')
        filename = os.path.splitext(self.filename)[0]

        if colormap == 'plasma':
            path=os.path.join(self.pathlist[0],filename + '.tif')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        elif colormap == 'Greys':
            path=os.path.join(self.pathlist[1],filename + '.jpg')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        elif colormap == 'BuPu':
            path=os.path.join(self.pathlist[2],filename + '.jpg')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        # plt.show()
    def rgb_depth(self,img):
        '''
        RGB-Depth Image
        MiDaS models for computing relative depth from a single image
        colormap = 'plasma'
        :param img: Image to be processed
        :return: numpy array
        '''
        # Load a model
        model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        midas = torch.hub.load("intel-isl/MiDaS", model_type)

        # Move model to GPU if available
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()

        # Load transforms to resize and normalize the image for large or small model
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        # Apply transforms
        input_batch = transform(img).to(device)

        # Predict and resize to original resolution
        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        self.print_image(output, 'plasma')

        return output
    def rgb_thermal(self,img):
        '''
        RGB-Thermal Image
        Convert original color image to grayscale
        colormap = 'Greys'
        :return:
        '''

        # RGB-T image(grayscale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.print_image(gray, 'Greys')

        return gray
    def rgb_event(self,img):
        '''
        RGB-Event Image
        Canny edge detection from OpenCV library
        colormap = 'BuPu'
        :param img:
        :return:
        '''
        # RGB-E image
        gray=self.rgb_thermal(img)
        canny = cv2.Canny(gray, 100, 200)

        self.print_image(canny,'BuPu')

        return canny
    def augment(self):
        images=os.listdir(self.path)
        for image in images:
            self.filename=image
            img=cv2.imread(self.path+"/"+image)

            # cv2 to plt
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.print_image(img, '')

            self.rgb_depth(img)
            self.rgb_thermal(img)
            self.rgb_event(img)

if(__name__ == "__main__"):
    parser=argparse.ArgumentParser(description='Choose one of animals. ')
    parser.add_argument('filepath',type=str,help='Path to image')
    args=parser.parse_args()

    aug=Augmentation(args.filepath)
    aug.augment()