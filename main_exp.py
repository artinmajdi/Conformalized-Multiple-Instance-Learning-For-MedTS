from models.conformalmil import ConformalMIL

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="ConformalMIL Training Script")
    
    parser.add_argument('--seed', type=int, default=2024, help='Random seed.')
    
    # Model parameters
    parser.add_argument('--in_features', type=int, default=1, help='Number of input features for the model.')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of output classes.')
    parser.add_argument('--embed', type=int, default=128, help='Embedding dimension size.')
    parser.add_argument('--seq_len', type=int, default=400, help='Maximum sequence length.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the model.')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.2, help='Learning rate for the optimizer.')
    parser.add_argument('--dropout_patch', type=float, default=0.1, help='Dropout rate for patch masking.')
    parser.add_argument('--cal_fraction', type=float, default=0.0, help='Fraction of training data for calibration.')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading.')

    # Miscellaneous parameters
    parser.add_argument('--dataset', type=str, default='default_dataset', help='Dataset name for loading data.')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to run the model on.')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Flag to use multiple GPUs for training.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    model = ConformalMIL(args)
    
    model.seed_everything(args.seed)
    
    #model.train()
    model.post_train_evaluation()