import torch.nn as nn
import tqdm
from torchvision.datasets import STL10
from torch.utils.data import DataLoader

def main():
    dataset = STL10(
        root=(Path().home() / 'datasets/STL10').as_posix(),
        download=True,
        split="train"
    )
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model, preprocess = get_clip_model()
    linear_probe = nn.Linear(512, len(labels))
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=3e-4)
    num_epochs = 10
    
    for epoch in range(num_epochs):
        pbar = tqdm.tqdm(enumerate(train_loader))
        epoch_loss: float = 0.0
        total_samples: int = 0
        for i, (image, label) in pbar:
            total_samples += label.shape[0]
            input_tensor = preprocess(image)
            # input_tensor = preprocess(image).unsqueeze(0)
            vision_embeddings = model.encode_image(input_tensor)
            optimizer.zero_grad()
            output_logits = linear_probe(vision_embeddings)
            output_logprob = F.log_softmax(output_logits, dim=-1)
            loss = F.nll_loss(output_logprob, label)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            pbar.set_description(f'Epoch {epoch} iteration {i}: loss {epoch_loss / float(total_samples)} ')


if __name__=='__main__':
    main()