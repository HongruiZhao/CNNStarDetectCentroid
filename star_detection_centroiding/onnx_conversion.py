import torch 

#model = torch.load("./saved_models/ELUnet_inter_2_90.pt").cpu()
#model = torch.load("./saved_models/MobileUNet_1.pt").cpu()
model = torch.load("./saved_models/straylight.pt").cpu()
images = torch.randn(1,1,224,224).cpu()

# output onnx 
torch.onnx.export(  model,               # model being run
                    images,                         # model input (or a tuple for multiple inputs)
                    "Unet_224.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    opset_version=12
                )
