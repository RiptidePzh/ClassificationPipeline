import torch
import wandb
from TrainingBackbone import *

if __name__ == '__main__':
    # Call GPU backend
    device = torch.device("mps" if torch.has_mps else "cpu")
    # Start Training Server (First time need to login into wandb)
    # TODO: Convert to local training server
    wandb.login()
    # Define config dictionary
    config = dict(
        epochs=30,
        classes=4,
        batch_size=32,
        learning_rate=0.005,
        dataset_dir='DataSet'
    )
    # Load a pretrained model
    from torchvision.models import densenet161, DenseNet161_Weights
    model = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1).to(device)
    # Config Training Pipeline
    pipeline = PytorchTrainingPipeline(model,config,device)
    # Starting Training Pipeline
    pipeline.model_pipeline(project_name='Deployed CV Model',
                            run_name='DenseNet 161')
    # Clean and Save Run
    wandb.finish()