import torch

cuda = torch.cuda.is_available()

print(cuda)

print(f"Your script has finished and the result is: {cuda}")

print(torch.cuda.device_count())

print(torch.cuda.current_device())


print(torch.cuda.device(0))

print(torch.cuda.get_device_name(0))
