from data.dataset import BreakHisDataset

def main():
    dataset = BreakHisDataset(
        root="/home-asustor/seddik/modelsineed/secondjournal/data_split_images_non_augmented",
        split="val",
        transform=None
    )

    img, label = dataset[0]
    print(type(img), label)
    print("Dataset length:", len(dataset))

if __name__ == "__main__":
    main()

    