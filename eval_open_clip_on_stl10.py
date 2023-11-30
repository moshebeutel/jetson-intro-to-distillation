from jetson_distillation.utils.openclip_utils import get_clip_model, get_clip_tokenizer
from jetson_distillation.utils.stl10_utils import (
    STL10_LABELS,
    get_stl10_transform,
    get_stl10_test_embedding_dataset,
    get_clip_stl10_text_embeddings
)
import torch.nn.functional as F
import tqdm
import torch
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from pathlib import Path

def embeddings_to_class_probs(vision_embeddings, text_embeddings):
    vision_embeddings = vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    logits = vision_embeddings @ text_embeddings.T
    class_probs = F.softmax(100. * logits, dim=-1)
    return class_probs

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'There sre {torch.cuda.device_count()} devices')

    print('Device properties',  torch.cuda.get_device_properties('cuda'))

    labels = STL10_LABELS
    print(labels)

    text_embeddings = get_clip_stl10_text_embeddings().to(device)
    print(f'encoded text embeddings for labels. Got tensor shape {text_embeddings.shape}')


    dataset = get_stl10_test_embedding_dataset()
    test_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    num_correct: int = 0
    total_samples: int = 0

    # pbar = tqdm.tqdm(test_loader)
    i = 0
    # for batch_image_embeddings, batch_labels in pbar:
    for _, batch_labels, batch_image_embeddings  in test_loader:
        i += 1
        total_samples += batch_labels.shape[0]
        batch_image_embeddings, batch_labels = batch_image_embeddings.to(device), batch_labels.to(device)
        output_class_probs = embeddings_to_class_probs(batch_image_embeddings, text_embeddings)
        output_labels = torch.argmax(output_class_probs, dim=-1)
        num_correct += int(torch.count_nonzero(output_labels == batch_labels))
        print(f'Iteration {i}: Correct: {num_correct}/{total_samples} ({100 * float(num_correct) / float(total_samples)}%)')
        # pbar.set_description(f'Iteration {i}: Correct: {num_correct}/{total_samples} ({100 * float(num_correct) / float(total_samples)}%)')


    accuracy = 100. * num_correct / len(dataset)

    print('Evaluation finished. Accuracy = ', accuracy)


if __name__=='__main__':
    main()