import torch
from torchvision.models import resnet18
import torch_pruning as tp
from neumeta.models import create_mnist_model
from neumeta.cifar10 import validate
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def setup_evaluation_data(batch_size=128):
    """
    Prepare the MNIST dataset for model evaluation.
    """
    # Data preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    val_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return val_loader, train_loader

def evaluate_model(model, data_loader, device):
    """
    Evaluate the performance of a model on the provided data.
    """
    # Preparation for model evaluation
    model.eval()  # Set the model to evaluation mode
    criterion = torch.nn.CrossEntropyLoss()

    # Perform validation and print results
    val_loss, acc = validate(model, data_loader, criterion, device)
    # print(f"Validation Loss: {val_loss:.4f}, Accuracy: {acc*100:.2f}%")
    return val_loss, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_batchs = 10
    # Prepare the validation data
    val_loader, train_loader = setup_evaluation_data()

    # Pruning setup
    example_inputs = torch.randn(1, 1, 28, 28).to(device)


    # Importance criteria
    imp_dict = {
        # 'Group Hessian': tp.importance.HessianImportance(group_reduction='mean'),
        # 'Group Taylor': tp.importance.TaylorImportance(group_reduction='mean'),
        # 'Group L1': tp.importance.MagnitudeImportance(p=1, group_reduction='mean'),
        # 'Group Slimming': tp.importance.BNScaleImportance(group_reduction='mean'),
        # 'Single-layer Slimming': tp.importance.BNScaleImportance(group_reduction='first'),
        'Random': tp.importance.RandomImportance(),
        'Hessian': tp.importance.HessianImportance(group_reduction='first'),
        'Taylor': tp.importance.TaylorImportance(group_reduction='first'),     
        'L1': tp.importance.MagnitudeImportance(p=1, group_reduction='first'),
        'L2': tp.importance.MagnitudeImportance(p=2, group_reduction="first"),   
    }
    # model_name = 'ResNet20'
    # target_layer = 'layer3.2.conv1'
    
    model_name = 'LeNet'
    # Specific layer to be pruned with varying degrees
    iterative_steps = 5
    for imp_name, imp in imp_dict.items():
        # Open a text file to record the results
        with open(f"pruning_results_{model_name}_{imp_name}_mnist.txt", "w") as results_file:
            # Write the header of the file
            results_file.write("Pruning Ratio,Resulting Channels,Validation Loss,Accuracy\n")


            # Experiment with different pruning ratios for the specific layer
            for pruning_ratio in [i * 0.05 for i in range(20)]:  # Adjust the range/sequence as needed
                # Reset the model before each pruning experiment
                model = create_mnist_model(model_name=model_name, hidden_dim=32).to(device)
                model.load_state_dict(torch.load(f"toy/mnist_{model.__class__.__name__}_dim{32}.pth"))

                # Define the pruning configuration
                pruning_config = {
                    'ignored_layers': [],  # Layers to exclude from pruning
                    # 'pruning_ratio': pruning_ratio,  # Specific pruning ratios per layer
                }
                for layer_name, layer_module in model.named_modules():
                    if layer_name.startswith('linear'):
                        pruning_config['ignored_layers'].append(layer_module)  # Exclude the final classifier

                # Initialize the pruner
                pruner = tp.pruner.MetaPruner(
                    model=model,
                    example_inputs=example_inputs,
                    importance=imp,
                    iterative_steps=iterative_steps,
                    pruning_ratio=pruning_ratio,
                    **pruning_config
                )

                for i in range(iterative_steps):
                    print(f"Pruning step {i+1}/{iterative_steps} with {imp_name} importance and {pruning_ratio * 100} pruning ratio:"  )
                    if isinstance(imp, tp.importance.HessianImportance):
                        # loss = F.cross_entropy(model(images), targets)
                        for k, (imgs, lbls) in enumerate(train_loader):
                            if k>=N_batchs: break
                            imgs = imgs.cuda()
                            lbls = lbls.cuda()
                            output = model(imgs) 
                            # compute loss for each sample
                            loss = torch.nn.functional.cross_entropy(output, lbls, reduction='none')
                            imp.zero_grad() # clear accumulated gradients
                            for l in loss:
                                model.zero_grad() # clear gradients
                                l.backward(retain_graph=True) # simgle-sample gradient
                                imp.accumulate_grad(model) # accumulate g^2
                    elif isinstance(imp, tp.importance.TaylorImportance):
                        # loss = F.cross_entropy(model(images), targets)
                        for k, (imgs, lbls) in enumerate(train_loader):
                            if k>=N_batchs: break
                            imgs = imgs.cuda()
                            lbls = lbls.cuda()
                            output = model(imgs)
                            loss = torch.nn.functional.cross_entropy(output, lbls)
                            loss.backward()
                    
                    # Execute the pruning
                    pruner.step()

                # Evaluate and display the model performance after pruning
                print(f"\nEvaluating model with pruned at {pruning_ratio * 100}%:")
                model.zero_grad()  # Clear any cached gradients
                # print(model)
                
                val_loss, acc = evaluate_model(model, val_loader, device)
                
                # Calculate the number of resulting channels
                resulting_channels = int(32 * (1 - pruning_ratio))  # Assuming 64 is the original number of channels

                # Record the results in the text file
                results_str = f"{pruning_ratio:.2f},{resulting_channels},{val_loss:.4f},{acc * 100:.2f}\n"
                results_file.write(results_str)

                # Optionally, print the results to the console
                print(f"Method: {imp_name}, Pruning ratio: {pruning_ratio:.2f}, Resulting Channels: {resulting_channels}, "
                      f"Validation Loss: {val_loss:.4f}, Accuracy: {acc * 100:.2f}")
                # exit()

# Entry point of the script
if __name__ == "__main__":
    main()
