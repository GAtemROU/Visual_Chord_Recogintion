import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def augment(root, savepath, id2label):
    original_dataset = ImageFolder(root)
    
    rotate_transform = transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.RandomRotation(30)
    ])

    rotate_crop_transform = transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.RandomRotation(30),
        transforms.RandomCrop((280, 210))
    ])

    color_transform = transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])

    rotate_crop_color_transform = transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.RandomRotation(30),
        transforms.RandomCrop((280, 210)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])

    rotate_crop_grayscale_transform = transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.RandomRotation(30),
        transforms.RandomCrop((280, 210)),
        transforms.Grayscale()
    ])

    rotate_color_transform = transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])

    for image_id, image_label_id in enumerate(original_dataset):
        image = image_label_id[0]
        label_id = image_label_id[1]
        for i in range(10):
            augmented_image = rotate_transform(image)
            augmented_image.save(f'{savepath}/{id2label[label_id]}/rotated_{image_id}_{i}.jpg')


    for image_id, image_label_id in enumerate(original_dataset):
        image = image_label_id[0]
        label_id = image_label_id[1]
        for i in range(10):
            augmented_image = rotate_crop_transform(image)
            augmented_image.save(f'{savepath}/{id2label[label_id]}/rotated_cropped_{image_id}_{i}.jpg')


    for image_id, image_label_id in enumerate(original_dataset):
        image = image_label_id[0]
        label_id = image_label_id[1]
        for i in range(10):
            augmented_image = color_transform(image)
            augmented_image.save(f'{savepath}/{id2label[label_id]}/colorJittered_{image_id}_{i}.jpg')


    for image_id, image_label_id in enumerate(original_dataset):
        image = image_label_id[0]
        label_id = image_label_id[1]
        for i in range(10):
            augmented_image = rotate_crop_color_transform(image)
            augmented_image.save(f'{savepath}/{id2label[label_id]}/rotated_cropped_colorJittered_{image_id}_{i}.jpg')


    for image_id, image_label_id in enumerate(original_dataset):
        image = image_label_id[0]
        label_id = image_label_id[1]
        for i in range(10):
            augmented_image = rotate_crop_grayscale_transform(image)
            augmented_image.save(f'{savepath}/{id2label[label_id]}/rotated_cropped_grayScaled_{image_id}_{i}.jpg')


    for image_id, image_label_id in enumerate(original_dataset):
        image = image_label_id[0]
        label_id = image_label_id[1]
        for i in range(10):
            augmented_image = rotate_color_transform(image)
            augmented_image.save(f'{savepath}/{id2label[label_id]}/rotated_colorJittered_{image_id}_{i}.jpg')