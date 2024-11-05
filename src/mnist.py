import numpy as np
import os.path

key_file = {
    "train_img": "train-images.idx3-ubyte",
    "train_label": "train-labels.idx1-ubyte",
    "test_img": "t10k-images.idx3-ubyte",
    "test_label": "t10k-labels.idx1-ubyte",
}


def read_idx3_ubyte(filename):
    # 바이너리 파일 열기
    with open(filename, "rb") as f:
        # 헤더 정보 읽기
        magic = int.from_bytes(f.read(4), "big")  # 매직 넘버
        n_images = int.from_bytes(f.read(4), "big")  # 이미지 개수
        n_rows = int.from_bytes(f.read(4), "big")  # 행 개수
        n_cols = int.from_bytes(f.read(4), "big")  # 열 개수

        # 이미지 데이터 읽기
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(n_images, n_rows, n_cols)

        return images


def _convert_numpy(dataset_dir):
    dataset = {}
    dataset["train_img"] = _load_img(dataset_dir + "/" + key_file["train_img"])
    dataset["train_label"] = _load_label(dataset_dir + "/" + key_file["train_label"])
    dataset["test_img"] = _load_img(dataset_dir + "/" + key_file["test_img"])
    dataset["test_label"] = _load_label(dataset_dir + "/" + key_file["test_label"])
    return dataset


def _load_label(file_name):
    with open(file_name, "rb") as f:
        # 헤더 정보 읽기
        magic = int.from_bytes(f.read(4), "big")  # 매직 넘버
        n_images = int.from_bytes(f.read(4), "big")  # 이미지 개수

        # 이미지 데이터 읽기
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    print(f"{file_name} {labels.shape} Done")
    return labels


def _load_img(file_name):
    with open(file_name, "rb") as f:
        # 헤더 정보 읽기
        magic = int.from_bytes(f.read(4), "big")  # 매직 넘버
        n_images = int.from_bytes(f.read(4), "big")  # 이미지 개수
        n_rows = int.from_bytes(f.read(4), "big")  # 행 개수
        n_cols = int.from_bytes(f.read(4), "big")  # 열 개수

        # 이미지 데이터 읽기
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(n_images, n_rows, n_cols)
    
    print(f"{file_name} {images.shape} Done")
    print("Done")
    return images


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(
    normalize=True, flatten=True, one_hot_label=False, dataset_dir="dataset/mnist"
):
    """
    Args:
        normalize (bool, optional): 입력 이미지의 픽셀 값을 0.0 ~ 1.0 사이의 값으로 정규화 or 0 ~ 255 사이 값. Defaults to True.
        flatten (bool, optional): "false로 설정하면 입력 이미지를 1 x 28 x 28 3차원 배열로. Defaults to True.
        one_hot_label (bool, optional): true이면 [0,0,1,0,0,0,0,0,0,0], false면 integer 2. Defaults to False.
        dataset_dir (str, optional): _description_. Defaults to "dataset/mnist".
    """
    dataset = _convert_numpy(dataset_dir)

    if normalize:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset["train_label"] = _change_one_hot_label(dataset["train_label"])
        dataset["test_label"] = _change_one_hot_label(dataset["test_label"])

    for key in ("train_img", "test_img"):
        if not flatten:
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
        else:
            dataset[key] = dataset[key].reshape(-1, 784)

    return (dataset["train_img"], dataset["train_label"]), (
        dataset["test_img"],
        dataset["test_label"],
    )
