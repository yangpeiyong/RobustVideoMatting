import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from model import MattingNetwork

# model = torchvision.models.mobilenet_v2(pretrained=True)
# model.eval()

model = MattingNetwork('mobilenetv3').eval()
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
model = model.eval()


scripted_module = torch.jit.script(model)
# Export full jit version model (not compatible mobile interpreter), leave it here for comparison
scripted_module.save("rvm_mobilenetv3.pt")
# Export mobile interpreter version model (compatible with mobile interpreter)

# optimized_scripted_module = optimize_for_mobile(scripted_module)
# optimized_scripted_module._save_for_lite_interpreter("rvm_mobilenetv3.ptl")

script_model_vulkan = optimize_for_mobile(scripted_module, backend='Vulkan')

script_model_vulkan._save_for_lite_interpreter("rvm_mobilenetv3_vulkan.ptl")


