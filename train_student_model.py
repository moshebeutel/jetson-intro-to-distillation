from jetson_distillation.utils.stl10_utils import (
    precompute_clip_stl10_train_image_embeddings,
    precompute_clip_stl10_test_image_embeddings,
    precompute_clip_stl10_text_embeddings,
    train_resnet18_from_scratch,
    train_resnet18_zero_shot_train_only,
    train_resnet18_linear_probe_train_only
)

precompute_clip_stl10_train_image_embeddings()
precompute_clip_stl10_test_image_embeddings()
precompute_clip_stl10_text_embeddings()
train_resnet18_from_scratch()
train_resnet18_zero_shot_train_only()
train_resnet18_linear_probe_train_only()