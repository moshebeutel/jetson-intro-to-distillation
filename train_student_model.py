from jetson_distillation.utils.stl10_utils import (
    train_resnet18_from_scratch,
    train_resnet18_zero_shot_train_only,
    train_resnet18_linear_probe_train_only
)

print('train_resnet18_from_scratch')
train_resnet18_from_scratch()
print('train_resnet18_zero_shot_train_only')
train_resnet18_zero_shot_train_only()
print('train_resnet18_linear_probe_train_only')
train_resnet18_linear_probe_train_only()