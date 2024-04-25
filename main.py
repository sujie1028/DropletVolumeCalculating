import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os

def to_volume(mask,real_dia=1):
    vol = 0
    drop_dia_idx = np.argmax(mask.sum((0,1)))
    drop_dia = int(mask[0,:,drop_dia_idx].sum())
    ratio = real_dia/drop_dia

    for w in range(mask.shape[2]):
        radius = mask[0,:,w].sum()/2 * ratio
        area = np.pi * radius**2
        height = 1 * ratio
        vol += height * area

    print(f"the true diamter of the droplet {real_dia} mm, The droplet pixel diameter is {drop_dia} pixels, The droplet volume is {vol:.4f} mm³ \n")
    return vol,drop_dia

def compute_vol(img_path,real_dia):
    print(f"Working on {img_path}")
    img_dir = 'raw_imgs'
    img_path = f'{img_dir}/{img_name}'

    # Load the model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # Set the model to evaluation mode
    model.eval()
    # Compute vol
    # compute_vol(img_path,real_dia)
    print(f"Working on {img_path}")
    # Load the image
    image = cv2.imread(img_path)

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to a PyTorch tensor
    tensor = torchvision.transforms.functional.to_tensor(image)

    # Add a batch dimension to the tensor
    tensor = tensor.unsqueeze(0)

    # Run the model
    with torch.no_grad():
        output = model(tensor)[0]

    # Get the boxes, labels, and masks
    boxes = output['boxes'].detach().numpy()
    labels = output['labels'].detach().numpy()
    masks = output['masks'].detach().numpy()
    scores = output['scores'].detach().cpu().numpy()

    # Choose a reliable mask to calculate the volume
    mask = masks[0]
    mask[mask<0.5] = 0
    mask[mask>=0.5] = 1
    mask_3d = np.repeat(mask, 3, axis=0).transpose(1,2,0)
    vol,drop_dia = to_volume(mask,real_dia)
    title_name = f"The drop's actual diameter is {real_dia} mm,\n The drop's pixel diameter is {drop_dia} pixels,\n the drop volume is {vol:.8f} mm³"
    if scores.max()<0.5:
        title_name = title_name + '\n Plz check the image, confidence is low!'

    # Plot the images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with two subplots
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Plot the first image in the first subplot
    ax[1].imshow(mask_3d, cmap='jet')  # Plot the second image in the second subplot
    ax[0].set_title(img_path)  # Set the title of the first subplot
    ax[1].set_title(f'Mask of {img_path}')  # Set the title of the second subplot
    ax[0].axis('off')
    ax[1].axis('off')
    fig.suptitle(title_name)
    plt.savefig(f'masked_imgs/Masked_{os.path.basename(img_path)}')
    plt.show()  # Display the figure

if __name__ == '__main__':
    img_name = '20G 1length 70Kpa of glycerol-P557-2.jpg'
    real_dia = 2.4222 # True droplet diameter unit: mm

    compute_vol(img_name,real_dia)

