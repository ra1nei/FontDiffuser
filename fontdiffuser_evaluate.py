import os
from torch.utils.data import DataLoader
from evaluator.evaluator import Evaluator
from evaluator.fid import FID
import torch
import torchvision.transforms as transforms
from PIL import Image


# -----------------------
# Dataset khớp với cấu trúc mới
# -----------------------
class SimpleResultDataset(torch.utils.data.Dataset):
    def __init__(self, result_dir, image_size=(96, 96)):
        self.result_dir = result_dir
        self.image_size = image_size

        # gom cặp generated/gt theo prefix (phần trước _generated_images)
        self.pairs = []
        all_files = os.listdir(result_dir)
        generated = [f for f in all_files if f.endswith("_generated_images.png")]

        for gen in generated:
            prefix = gen.replace("_generated_images.png", "")
            gt = prefix + "_gt_images.png"
            gt_path = os.path.join(result_dir, gt)
            gen_path = os.path.join(result_dir, gen)
            if os.path.exists(gt_path):
                self.pairs.append((gt_path, gen_path, prefix))

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gt_path, gen_path, key = self.pairs[idx]
        gt = self.transform(Image.open(gt_path).convert("L"))
        gen = self.transform(Image.open(gen_path).convert("L"))
        return (gt.unsqueeze(0), gen.unsqueeze(0), key)


# -----------------------
# Run evaluation
# -----------------------
def evaluate(result_dir, evaluate_mode="style", gpu_ids=[0]):
    # Fake opt object để tương thích Evaluator
    class Opt:
        def __init__(self):
            self.evaluate_mode = evaluate_mode
            self.results_dir = os.path.dirname(result_dir)
            self.name = os.path.basename(result_dir)
            self.phase = "test"
            self.epoch = "latest"
            self.gpu_ids = gpu_ids

    opt = Opt()
    dataset = SimpleResultDataset(result_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # khởi tạo Evaluator (không cần classifier)
    evaluator = Evaluator(opt, num_classes=1, text2label={})

    for data in dataloader:
        evaluator.evaluate(data)
        evaluator.record_current_results()

    evaluator.compute_final_results()


if __name__ == "__main__":
    from configs.fontdiffuser import get_parser
    parser = get_parser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    result_folder = f"results/{args.model}-{args.name}"
    evaluate(result_folder)