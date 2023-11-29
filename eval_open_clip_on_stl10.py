from jetson_distillation.utils.openclip_utils import get_clip_model, get_clip_tokenizer
from jetson_distillation.utils.stl10_utils import STL10_LABELS, get_stl10_transform
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
    model, preprocess = get_clip_model()
    tokenizer = get_clip_tokenizer()
    labels = STL10_LABELS
    print(labels)

    text = tokenizer(labels)
    text_embeddings = model.encode_text(text)
    print(f'encoded text embeddings for labels. Got tensor shape {text_embeddings.shape}')


    dataset = STL10(
        root=(Path().home() / 'datasets/STL10').as_posix(),
        download=True,
        split="test",
        transform=get_stl10_transform()
    )
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    num_correct: int = 0
    total_samples: int = 0

    pbar = tqdm.tqdm(enumerate(test_loader))
    for i, (image, label) in pbar:
        # input_tensor = preprocess(image).unsqueeze(0)
        total_samples += label.shape[0]
        input_tensor = preprocess(image)
        vision_embeddings = model.encode_image(input_tensor)
        output_class_probs = embeddings_to_class_probs(vision_embeddings, text_embeddings)
        output_label = torch.argmax(output_class_probs, dim=-1)
        num_correct += int(torch.count_nonzero(output_label == label))
        pbar.set_description(f'Iteration {i}: Correct: {num_correct}/{total_samples} ({100 * float(num_correct) / float(total_samples)}%)')


    accuracy = 100. * num_correct / len(dataset)





if __name__=='__main__':
    main()