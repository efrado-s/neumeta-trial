from torchvision import transforms, datasets
from torch.utils.data import DataLoader, DistributedSampler

def get_cifar10(batch_size, strong_transform=False):
    """
    Returns train and validation data loaders for CIFAR-10 dataset.

    Args:
        batch_size (int): Number of samples per batch to load.
        strong_transform (bool): Decides whether strong_transform is performed to the dataset or not.

    Returns:
        tuple: A tuple of train and validation data loaders.
    """

    # Data preparation
    # Transforming train dataset
    if strong_transform == 'v1':
        print("Using strong transform v1")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, shear=15, scale=(0.8,1.2)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.201])]
            )
    elif strong_transform == 'v2':
        print("Using strong transform v2")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.201])]
            )
    elif strong_transform == 'v3':
        print("Using strong transform v2")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.201])]
            )
    else:
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.201])
    ])
    
    # Transforming test dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.201])
    ])

    # Get dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_mnist(batch_size, strong_transform=False):
    # Data preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    val_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def get_dataset(dataset_name, batch_size, strong_transform=False):
    print(f'Using dataset: {dataset_name} with batch size: {batch_size} and strong transform: {strong_transform}')
    
    if dataset_name == 'cifar10':
        train_loader, val_loader = get_cifar10(batch_size, strong_transform)
    elif dataset_name == 'mnist':
        train_loader, val_loader =  get_mnist(batch_size, strong_transform)
    
    return train_loader, val_loader