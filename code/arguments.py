# args
args_shuffled_batching = {
    'embedding_dim': 300,
    'hidden_dim': 300,
    'num_layers': 2,
    'dropout_prob': 0.5,
    'shuffle': True,
    'reuse_hidden': False,
    'tie_weight': False,
    'lr': 0.001,
    'gamma': 1.00,
    'epochs': 30,
    'batch_size': 32,
    'sequence_length': 10,
    'save_path': 'assets/'
}

args_continuous_batching = {
    'embedding_dim': 300,
    'hidden_dim': 300,
    'num_layers': 2,
    'dropout_prob': 0.5,
    'shuffle': False,
    'reuse_hidden': False,
    'tie_weight': False,
    'lr': 0.001,
    'gamma': 1.00,
    'epochs': 30,
    'batch_size': 32,
    'sequence_length': 10,
    'save_path': 'assets/'
}

args_continuous_batching_reuse = {
    'embedding_dim': 300,
    'hidden_dim': 300,
    'num_layers': 2,
    'dropout_prob': 0.5,
    'shuffle': False,
    'reuse_hidden': True,
    'tie_weight': False,
    'lr': 0.001,
    'gamma': 1.00,
    'epochs': 30,
    'batch_size': 32,
    'sequence_length': 10,
    'save_path': 'assets/'
}

args_continuous_batching_reuse_rnn = {
    'embedding_dim': 300,
    'hidden_dim': 300,
    'num_layers': 2,
    'dropout_prob': 0.5,
    'shuffle': False,
    'reuse_hidden': True,
    'tie_weight': False,
    'lr': 0.001,
    'gamma': 1.00,
    'epochs': 30,
    'batch_size': 32,
    'sequence_length': 10,
    'save_path': 'assets/'
}

args_dropout_len7 = {
    'embedding_dim': 300,
    'hidden_dim': 300,
    'num_layers': 2,
    'dropout_prob': 0.5,
    'shuffle': False,
    'reuse_hidden': True,
    'tie_weight': False,
    'lr': 0.001,
    'gamma': 1.00,
    'epochs': 30,
    'batch_size': 32,
    'sequence_length': 7,
    'save_path': 'assets/'
}

args_layers2 = {
    'embedding_dim': 300,
    'hidden_dim': 300,
    'num_layers': 2,
    'dropout_prob': 0.5,
    'shuffle': False,
    'reuse_hidden': True,
    'tie_weight': False,
    'lr': 0.001,
    'gamma': 1.00,
    'epochs': 30,
    'batch_size': 32,
    'sequence_length': 7,
    'save_path': 'assets/'
}

args_layers3 = {
    'embedding_dim': 300,
    'hidden_dim': 300,
    'num_layers': 3,
    'dropout_prob': 0.5,
    'shuffle': False,
    'reuse_hidden': True,
    'tie_weight': False,
    'lr': 0.001,
    'gamma': 1.00,
    'epochs': 30,
    'batch_size': 32,
    'sequence_length': 7,
    'save_path': 'assets/'
}

args_layers4 = {
    'embedding_dim': 300,
    'hidden_dim': 300,
    'num_layers': 4,
    'dropout_prob': 0.5,
    'shuffle': False,
    'reuse_hidden': True,
    'tie_weight': False,
    'lr': 0.001,
    'gamma': 1.00,
    'epochs': 30,
    'batch_size': 32,
    'sequence_length': 7,
    'save_path': 'assets/'
}

args_weight_tying = {
    'embedding_dim': 300,
    'hidden_dim': 300,
    'num_layers': 2,
    'dropout_prob': 0.5,
    'shuffle': False,
    'reuse_hidden': True,
    'tie_weight': True,
    'lr': 0.001,
    'gamma': 1.00,
    'epochs': 30,
    'batch_size': 32,
    'sequence_length': 7,
    'save_path': 'assets/'
}

args_lr_decay01 = {
    'embedding_dim': 300,
    'hidden_dim': 300,
    'num_layers': 2,
    'dropout_prob': 0.5,
    'shuffle': False,
    'reuse_hidden': True,
    'tie_weight': True,
    'lr': 0.001,
    'gamma': 0.1,
    'epochs': 30,
    'batch_size': 32,
    'sequence_length': 7,
    'save_path': 'assets/'
}

args_lr_decay05 = {
    'embedding_dim': 300,
    'hidden_dim': 300,
    'num_layers': 2,
    'dropout_prob': 0.5,
    'shuffle': False,
    'reuse_hidden': True,
    'tie_weight': True,
    'lr': 0.001,
    'gamma': 0.5,
    'epochs': 30,
    'batch_size': 32,
    'sequence_length': 7,
    'save_path': 'assets/'
}

args_lr_decay098 = {
    'embedding_dim': 300,
    'hidden_dim': 300,
    'num_layers': 2,
    'dropout_prob': 0.5,
    'shuffle': False,
    'reuse_hidden': True,
    'tie_weight': True,
    'lr': 0.001,
    'gamma': 0.98,
    'epochs': 30,
    'batch_size': 32,
    'sequence_length': 7,
    'save_path': 'assets/'
}