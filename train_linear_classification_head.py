import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from eval_open_clip_on_stl10 import embeddings_to_class_probs
from jetson_distillation.utils.stl10_utils import (
    STL10_LABELS,
    get_clip_stl10_text_embeddings,
    get_stl10_train_embedding_dataset,
    get_stl10_test_embedding_dataset
)

def main():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'There sre {torch.cuda.device_count()} devices')

    print('Device properties',  torch.cuda.get_device_properties('cuda'))

    labels = STL10_LABELS
    print(labels)

    trainset = get_stl10_train_embedding_dataset()
    train_loader = DataLoader(trainset, batch_size=2, shuffle=True)
    total_samples: int = 0
    linear_probe = nn.Linear(512, len(labels)).to(device)
    linear_probe.train()
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=3e-4)
    num_epochs = 10
    
    for epoch in range(num_epochs):
        pbar = tqdm.tqdm(enumerate(train_loader))
        epoch_loss: float = 0.0
        relative_epoch_loss: str = ''
        total_samples: int = 0
        for i, (_, batch_labels, batch_image_embeddings) in pbar:
            total_samples += batch_labels.shape[0]
            batch_image_embeddings, batch_labels = batch_image_embeddings.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            output_logits = linear_probe(batch_image_embeddings)
            output_logprob = F.log_softmax(output_logits, dim=-1)
            loss = F.nll_loss(output_logprob, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            relative_epoch_loss = '%.3f' % (epoch_loss / float(total_samples))
            pbar.set_description(f'Epoch {epoch} iteration {i}: loss {relative_epoch_loss} ')
        torch.save(linear_probe.state_dict(), f'linear_probe_loss_{relative_epoch_loss}.pt')


    # linear_probe = nn.Linear(512, len(labels)).to(device)
    # linear_probe.load_state_dict(torch.load((Path.home() / 'saved_models/linear_prob_loss_25.pt').as_posix()))
    linear_probe.eval()
    num_correct: int = 0
    total_samples: int = 0
    testset = get_stl10_test_embedding_dataset()
    test_loader = DataLoader(testset, batch_size=2, shuffle=False)
    pbar = tqdm.tqdm(enumerate(test_loader))
    for i, (_, batch_labels, batch_image_embeddings) in pbar:
        total_samples += batch_labels.shape[0]
        batch_image_embeddings, batch_labels = batch_image_embeddings.to(device), batch_labels.to(device)
        output_logits = linear_probe(batch_image_embeddings)
        output_logprob = F.log_softmax(output_logits, dim=-1)
        # output_class_probs = embeddings_to_class_probs(batch_image_embeddings, text_embeddings)
        output_labels = torch.argmax(output_logprob, dim=-1)
        num_correct += int(torch.count_nonzero(output_labels == batch_labels))
        pbar.set_description(f'Iteration {i}: Correct: {num_correct}/{total_samples} ({100 * float(num_correct) / float(total_samples)}%)')
    accuracy = 100. * num_correct / len(testset)
    print('Evaluation finished. Accuracy = ', accuracy)

if __name__=='__main__':
    main()