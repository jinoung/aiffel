'''
Convolutional Neural Network: Scissor-Rock-Paper Classifier
1. Package load
2. 데이터셋 다운로드 및 훈련, 검증, 테스트 데이터셋 구성
3. 하이퍼파라미터 세팅 : batch_size, num_epochs, learning_rate
4. Dataset 및 DataLoader 할당 : ScissorRockPaperDataset, train_loader = DataLoader(...)
5. 네트워크 설계 : class SimpleCNN(nn.Module):
6. train, validation, test 함수 정의
7. 모델 저장 함수 정의 : def save_model(model, saved_dir, file_name='best_model.pt'):
8. 모델 생성 및 Loss function, Optimizer 정의 : model = SimpleCNN(), criterion=, optimizer=
9. Training : train(...)
10. 저장된 모델 불러오기 및 test : model.load_state_dict(checkpoint['net']), test(...)
'''

import os
from pathlib import Path

# 프로젝트를 저장한 디렉토리를 입력하세요!
project_dir = "aiffel"

base_path = Path("C:/DeepLearning/")
project_path = base_path / project_dir
os.chdir(project_path)
for x in list(project_path.glob("*")):
    if x.is_dir():
        dir_name = str(x.relative_to(project_path))
        os.rename(dir_name, dir_name.split(" ", 1)[0])
print(f"현재 디렉토리 위치: {os.getcwd()}")

import torch
print('pytorch version: {}'.format(torch.__version__))

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline

print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))
device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 사용 가능 여부에 따라 device 정보 저장

import zipfile
from pathlib import Path

current_path = Path().absolute()
data_path = current_path / "data"
print("현재 디렉토리 위치: {}".format(current_path))
if (data_path / "ex1").exists():
    print("이미 'data/ex1' 폴더에 압축이 풀려있습니다. 확인해보세요!")
else:
    with zipfile.ZipFile(str(data_path / "ex1.zip"), "r") as zip_ref:
        zip_ref.extractall(str(data_path / "ex1"))
    print("Done!")

data_dir = './data/ex1'  # 압축 해제된 데이터셋의 디렉토리 경로

#하이퍼파라미터 세팅
batch_size = 100
num_epochs = 10
learning_rate = 0.0001


class ScissorRockPaperDataset(Dataset):
    def __init__(self, data_dir, mode, transform=None):
        self.all_data = sorted(glob.glob(os.path.join(data_dir, mode, '*', '*')))
        self.transform = transform

    def __getitem__(self, index):
        # Step 1: 반환할 이미지 경로 정의 및 이미지 로드
        data_path = self.all_data[index]
        img = Image.open(data_path)
        if self.transform is not None:
            img = self.transform(img)

        # Step 2: 이미지에 대한 label 정의
        if os.path.basename(data_path).startswith('scissor'):
            label = 0
        elif os.path.basename(data_path).startswith('rock'):
            label = 1
        elif os.path.basename(data_path).startswith('paper'):
            label = 2
        else:
            print('error')

        return img, label

    def __len__(self):
        length = len(self.all_data)
        return length

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(120, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([120, 120]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#Dataset 및 DataLoader 할당
train_data = ScissorRockPaperDataset(data_dir='./data/ex1', mode='train', transform=data_transforms['train'])
val_data = ScissorRockPaperDataset(data_dir='./data/ex1', mode='val', transform=data_transforms['val'])
test_data = ScissorRockPaperDataset(data_dir='./data/ex1', mode='test', transform=data_transforms['val'])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

#네트워크 설계
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # 첫번째층
            # ImgIn shape= (?, 224, 224, 3)
            #    Conv     -> (?, 222, 222, 32)
            #    Pool     -> (?, 111, 111, 32)
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),  # conv_1 해당하는 층
            torch.nn.BatchNorm2d(32),  # batch_norm_1 해당하는 층
            torch.nn.ReLU(),  # ReLU_1 해당하는 층
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # maxpool_1 해당하는 층
            # 두번째층
            # ImgIn shape= (?, 111, 111, 32)
            #    Conv     -> (?, 109, 109, 64)
            #    Pool     -> (?, 54, 54, 64)
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),  # conv_2 해당하는 층
            torch.nn.BatchNorm2d(64),  # batch_norm_2 해당하는 층
            torch.nn.ReLU(),  # ReLU_2 해당하는 층
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # maxpool_2 해당하는 층
            # 세번째층
            # ImgIn shape=(?, 28, 28, 64) (?, 54, 54, 64)
            #    Conv     -> (?, 52, 52, 128)
            #    Pool     -> (?, 26, 26, 128)
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),  # conv_3 해당하는 층
            torch.nn.BatchNorm2d(128),  # batch_norm_3 해당하는 층
            torch.nn.ReLU(),  # ReLU_3 해당하는 층
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # maxpool_3 해당하는 층
            # 네번째층
            # ImgIn shape= (?, 26, 26, 128)
            #    Conv     -> (?, 24, 24, 128)
            #    Pool     -> (?, 12, 12, 128)
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),  # conv_4 해당하는 층
            torch.nn.BatchNorm2d(128),  # batch_norm_4 해당하는 층
            torch.nn.ReLU(),  # ReLU_4 해당하는 층
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # maxpool_4 해당하는 층
        )

        # self.fc 구현
        self.fc1 = torch.nn.Linear(128 * 12 * 12, 512, bias=True)
        self.fc2 = torch.nn.Linear(512, 2, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#train, validation, test 함수 정의
def train(num_epochs, model, data_loader, criterion, optimizer, saved_dir, val_every, device):
    print('Start training..')
    best_loss = 9999999
    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax).float().mean()

            if (i+1) % 3 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(
                    epoch+1, num_epochs, i+1, len(data_loader), loss.item(), accuracy.item() * 100))

        if (epoch + 1) % val_every == 0:
            avrg_loss = validation(epoch + 1, model, val_loader, criterion, device)
            if avrg_loss < best_loss:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_loss = avrg_loss
                save_model(model, saved_dir)

def validation(epoch, model, data_loader, criterion, device):
    print('Start validation #{}'.format(epoch) )
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        for i, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total += imgs.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss
            cnt += 1
        avrg_loss = total_loss / cnt
        print('Validation #{}  Accuracy: {:.2f}%  Average Loss: {:.4f}'.format(epoch, correct / total * 100, avrg_loss))
    model.train()
    return avrg_loss

def test(model, data_loader, device):
    print('Start test..')
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            ## 코드 시작 ##
            outputs = model(imgs)
            ## 코드 종료 ##
            _, argmax = torch.max(outputs, 1)    # max()를 통해 최종 출력이 가장 높은 class 선택
            total += imgs.size(0)
            correct += (labels == argmax).sum().item()

        print('Test accuracy for {} images: {:.2f}%'.format(total, correct / total * 100))
    model.train()

#모델 저장 함수 정의
def save_model(model, saved_dir, file_name='best_model.pt'):
    os.makedirs(saved_dir, exist_ok=True)
    check_point = {
        'net': model.state_dict()
    }
    output_path = os.path.join(saved_dir, file_name)
    torch.save(check_point, output_path)

#모델 생성 및 Loss function, Optimizer 정의
torch.manual_seed(7777) # 일관된 weight initialization을 위한 random seed 설정
model = SimpleCNN()
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

model = model.to(device)
val_every = 1
saved_dir = './saved/SimpleCNN'

# Training
train(num_epochs, model, train_loader, criterion, optimizer, saved_dir, val_every, device)

# 저장된 모델 불러오기 및 test
model_path = './saved/SimpleCNN/best_model.pt'
model = SimpleCNN().to(device)   # 아래의 모델 불러오기를 정확히 구현했는지 확인하기 위해 새로 모델을 선언하여 학습 이전 상태로 초기화

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['net'])

test(model, test_loader, device)
'''
##학습된 모델의 예측 결과를 시각화
columns = 5
rows = 5
fig = plt.figure(figsize=(8, 8))

model.eval()
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
for i in range(1, columns * rows + 1):
    data_idx = np.random.randint(len(test_data))
    input_img = test_data[data_idx][0].unsqueeze(dim=0).to(device)
    '''
    unsqueeze()를 통해 입력 이미지의 shape을 (1, 28, 28)에서 (1, 1, 28, 28)로 변환. 
    모델에 들어가는 입력 이미지의 shape은 (batch_size, channel, width, height) 되어야 함에 주의하세요!
    '''
    output = model(input_img)
    _, argmax = torch.max(output, 1)
    pred = argmax.item()
    label = test_data[data_idx][1]

    fig.add_subplot(rows, columns, i)
    pred_title = 'Cat' if pred == 0 else 'Dog'
    if pred == label:
        plt.title(pred_title + '(O)')
    else:
        plt.title(pred_title + '(X)')
    plot_img = test_data[data_idx][0]
    # 이미지를 normalization 이전 상태로 되돌리는 작업
    plot_img[0, :, :] = plot_img[0, :, :] * std[0] + mean[0]
    plot_img[1, :, :] = plot_img[1, :, :] * std[1] + mean[1]
    plot_img[2, :, :] = plot_img[2, :, :] * std[2] + mean[2]
    plot_img = transforms.functional.to_pil_image(plot_img)
    plt.imshow(plot_img)
    plt.axis('off')
plt.show()
'''