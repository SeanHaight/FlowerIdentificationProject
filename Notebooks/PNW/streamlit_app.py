import streamlit as st # type: ignore
from torchvision.transforms import v2 #type: ignore
from PIL import Image

#Some of our images are .pngs, we drop the alpha. 
train_transform = v2.Compose([
    #Rely on the v2 transforms from torchvision.transforms.v2

    #Use tensors instead of PIL images

    #Use torch.uint8 dtype, especially for resizing
                                v2.ToPILImage(),
                                v2.ToTensor(),
                                v2.RandomAffine(degrees=(-180,180), translate=(0,.1), scale=(.9,1)),
                                v2.RandomHorizontalFlip(p=0.5),
                                v2.RandomVerticalFlip(p=0.5),
                                v2.ColorJitter(brightness=.3, hue=.01),
                                v2.Resize ( (256,256) , interpolation=2 ),
                                v2.CenterCrop(224),
                                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                            ])
val_transform = v2.Compose([
                                v2.ToPILImage(),
                                v2.ToTensor(),
                                v2.Resize ( (256,256) , interpolation=2 ),
                                v2.CenterCrop(224),
                                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
target_transform = v2.Compose([
                                v2.Lambda(lambda x: torch.tensor(x).long()),
                            ])

import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch
import torch.nn as nn

class SeanBlock(BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(SeanBlock, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        self.dropout_prob = .5
        
    def forward(self,x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.Dropout(self.dropout_prob)(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
    
class SeanBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(SeanBottleneck, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        self.dropout_prob = .375
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.Dropout(self.dropout_prob)(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.Dropout(self.dropout_prob)(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class MyResNet50(ResNet):
    def __init__(self):
        super(MyResNet50, self).__init__(SeanBottleneck, [3, 4, 6, 3])
        
    def forward(self, x):
        # change forward here
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model = MyResNet50()
# if you need pretrained weights
model.load_state_dict(torch.load("../../Models/PNW/resnet50midtrain.pt"))
model.eval()
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Title
st.title('Image Classification with PyTorch and Streamlit')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image = val_transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    # Define class names
    class_names = ['Not Flowering', 'Flowering']  # Replace with your actual class names

    st.write(f'Predicted: {class_names[predicted]}')