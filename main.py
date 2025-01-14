import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

from models.vit import ViTBinaryClassifier
from models.cnn import SimpleCNN

from torch.optim.lr_scheduler import StepLR

def parse_label_from_filename(filename: str):
    """
    ファイル名から right or left のラベル(2値)を返す。
      - right -> 1
      - left  -> 0
      - それ以外 (baseやstep_xなど) -> None
    """
    lower_name = filename.lower()
    
    # "base" を含むファイルは除外
    if "base" in lower_name:
        return None
    
    # "step_1" ~ "step_5" を含むファイルは除外
    # 1~5それぞれをチェックする
    for i in range(1, 6):
        if f"step_{i}" in lower_name:
            return None

    # "right", "left" の判定
    if "right" in lower_name:
        return 1
    elif "left" in lower_name:
        return 0
    else:
        return None


class ImageDataset(Dataset):
    def __init__(self, file_paths, transform=None, skip_no_label=True):
        """
        Args:
            file_paths (list[str]): 画像ファイルへのパス一覧
            transform (callable, optional): 画像に適用する変換処理
            skip_no_label (bool): baseなどラベル付けをしないものをスキップするかどうか
        """
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for path in file_paths:
            filename = os.path.basename(path)
            label = parse_label_from_filename(filename)
            if label is None and skip_no_label:
                continue
            self.image_paths.append(path)
            self.labels.append(label)

        if len(self.image_paths) == 0:
            raise ValueError("No valid image files found (or all were skipped).")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # 画像読み込み
        image = Image.open(image_path)
        # 必要に応じてRGB変換
        image = image.convert('RGB')
        # グレースケールに変換
        # image = image.convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    # ===========================================================
    # 1. ディレクトリ内の全ファイルを取得 & シャッフル & Train/Val/Test 分割
    # ===========================================================
    data_dir = "./original_images"  # ここに全画像が入ったディレクトリを指定
    all_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]

    # 再現性のため乱数シード設定（任意）
    random.seed(42)
    random.shuffle(all_files)

    # Split割合（例: 8:1:1）
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    num_total = len(all_files)
    train_end = int(train_ratio * num_total)
    val_end = int((train_ratio + val_ratio) * num_total)

    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    print(f"Total: {num_total} images")
    print(f"Train: {len(train_files)} images")
    print(f"Val:   {len(val_files)} images")
    print(f"Test:  {len(test_files)} images")

    # ===========================================================
    # 2. Dataset, DataLoader 作成 (Train, Val, Test)
    # ===========================================================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageDataset(train_files, transform=transform, skip_no_label=True)
    val_dataset   = ImageDataset(val_files,   transform=transform, skip_no_label=True)
    test_dataset  = ImageDataset(test_files,  transform=transform, skip_no_label=True)
    
    # check size
    print(f"Train: {len(train_dataset)} images")
    print(f"Val:   {len(val_dataset)} images")
    print(f"Test:  {len(test_dataset)} images")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=2)

    # ===========================================================
    # 3. モデル定義, 損失関数, Optimizer
    # ===========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = SimpleCNN(num_classes=2).to(device)
    model = ViTBinaryClassifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # ===========================================================
    # 4. 学習ループ (Train)
    # ===========================================================
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Iter [{i+1}/{len(train_loader)}] "
                      f"Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

        # -------------------------------------------------------
        # 5. エポックごとに Validation で性能をチェックする例
        # -------------------------------------------------------
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.long().to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100.0 * correct / total if total > 0 else 0
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        scheduler.step()

    print("Training Finished!")

    # ===========================================================
    # 6. テストデータで最終評価 (Test)
    # ===========================================================
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.long().to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100.0 * correct / total if total > 0 else 0
    print(f"Test Accuracy: {test_acc:.2f}%")
