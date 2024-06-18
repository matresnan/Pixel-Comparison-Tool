import subprocess
import timeit
import numpy as np

from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


class ImageNumberError(Exception):
    """自定义异常类，处理图片数量少于2的情况。"""

    pass


class ImageSizeError(Exception):
    """自定义异常类，处理图片大小不一致的情况。"""

    pass


def timer_decorator(func):
    """装饰器，用于测量函数执行时间。"""

    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.2f} 秒")
        return result

    return wrapper


def get_all_png(folder: Path) -> list:
    """从文件夹中获取所有的PNG文件。"""
    return list(folder.glob("*.png"))


def load_image(path: Path) -> np.ndarray:
    """加载图像并将其转换为NumPy数组。"""
    return np.array(Image.open(path).convert("RGBA"))


@timer_decorator
def load_and_check_images(images: list[Path]) -> list[np.ndarray]:
    """加载图像并检查它们的大小是否一致。"""
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_image, image) for image in images]

        loaded_images = []
        first_size = None

        for future in as_completed(futures):
            img = future.result()
            if first_size is None:
                first_size = img.shape
            elif img.shape != first_size:
                raise ImageSizeError("图像大小不一致！")
            loaded_images.append(img)
    return loaded_images


@timer_decorator
def compare_pixels(
    first_pixels: np.ndarray, other_pixels_list: list[np.ndarray]
) -> Image.Image:
    """比较第一张图像与其他图像的像素，并创建一个差异掩码。"""
    diff_mask = np.zeros(first_pixels[:, :, 0].shape, dtype=np.uint8)

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(compare_pixels_with_other, first_pixels, other_pixels)
            for other_pixels in other_pixels_list
        ]
        for future in as_completed(futures):
            diff_mask |= future.result()

    first_pixels[diff_mask.nonzero()] = [0, 0, 0, 0]

    if first_pixels.dtype != np.uint8:
        first_pixels = first_pixels.astype(np.uint8)
    return Image.fromarray(first_pixels)


def compare_pixels_with_other(first_pixels, other_pixels):
    """比较两张图像的像素并返回一个差异的布尔掩码。"""
    return np.any(first_pixels != other_pixels, axis=-1).astype(np.uint8)


@timer_decorator
def main(folder: Path, flag: bool):
    """主函数，用于加载和比较图像。"""
    out_file_path = folder / "same_pixel.png"
    files = get_all_png(folder)
    count = len(files)
    if count < 2:
        raise ImageNumberError("图片数量少于2！")

    pixels_list = load_and_check_images(files)
    same_image = compare_pixels(pixels_list[0], pixels_list[1:])
    same_image.save(out_file_path)

    if flag:
        subprocess.run(["explorer", "/select,", str(out_file_path)])


if __name__ == "__main__":
    print("PIL版本:", Image.__version__)
    src_folder = r"C:\Users\admin\Desktop\iPhone6\升级提醒"
    folder_path = Path(src_folder).resolve()
    if not folder_path.exists():
        raise FileNotFoundError(f"{folder_path} 不存在！")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"{folder_path} 不是一个文件夹！")
    print("待处理的路径:", folder_path)
    main(folder_path, False)
