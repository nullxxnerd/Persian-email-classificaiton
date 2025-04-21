import torch
import pickle

# This script saves the trained TextCNN model and associated tokenizer and config.
# Run this after training completes and the following variables are available in your namespace:
# `trained_model`, `tokenizer`, `max_len`, `label_map_inv`, `stop_words`.

# Save model state_dict
torch.save(trained_model.state_dict(), 'textcnn.pth')

# Save tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Save configuration (preprocessing & model architecture)
model_config = {
    'vocab_size': trained_model.embedding.num_embeddings,
    'embedding_dim': trained_model.embedding.embedding_dim,
    'num_filters': trained_model.convs[0].out_channels,
    'filter_sizes': [conv.kernel_size[0] for conv in trained_model.convs],
    'num_classes': trained_model.fc.out_features,
    'dropout': trained_model.dropout.p
}
config = {
    'max_len': max_len,
    'label_map_inv': label_map_inv,
    'stop_words': stop_words,
    'model_config': model_config
}
with open('config.pkl', 'wb') as f:
    pickle.dump(config, f)

print('Export complete: textcnn.pth, tokenizer.pkl, config.pkl')
