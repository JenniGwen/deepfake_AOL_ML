import torch
import torch.nn as nn
import timm

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        old_conv = self.backbone.conv_stem
        new_conv = nn.Conv2d(4, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=old_conv.bias is not None)
        self.backbone.conv_stem = new_conv
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(1280, 256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, 1))
    def forward(self, x):
        return self.backbone(x)

model = DeepfakeDetector()
state = torch.load("best_model.pth", map_location="cpu", weights_only=False)

print(f"Keys in checkpoint: {len(state)}")
print(f"Keys in model:      {len(model.state_dict())}")

result = model.load_state_dict(state, strict=False)
print(f"\nMissing keys (not loaded, stay random!): {len(result.missing_keys)}")
for k in result.missing_keys[:20]:
    print(f"  {k}")

print(f"\nUnexpected keys (in file but unused): {len(result.unexpected_keys)}")
for k in result.unexpected_keys[:20]:
    print(f"  {k}")

# Check sample tensor for sanity
ckpt_first = list(state.keys())[0]
model_first = list(model.state_dict().keys())[0]
print(f"\nCheckpoint first key: {ckpt_first}  shape={state[ckpt_first].shape}")
print(f"Model first key:      {model_first}  shape={model.state_dict()[model_first].shape}")