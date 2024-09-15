import torch
from pathlib import Path
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter

from .utils import cycle
from torch.optim.lr_scheduler import LambdaLR
from ema_pytorch import EMA
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from math import sqrt
from process_data import cal_mean_std


def batch_psnr(imgs1, imgs2):
    with torch.no_grad():
        batch_mse = torch.mean((imgs1 - imgs2) ** 2, axis=(1, 2, 3))
        batch_psnr = 20 * torch.log10(1.0 / torch.sqrt(batch_mse))
        return torch.mean(batch_psnr)

# trainer class
class Trainer(object):
    def __init__(
        self,
        rank,
        sample_steps,
        diffusion_model,
        train_dl,
        val_dl,
        scheduler_function,
        train_lr=1e-4,
        train_num_steps=1000000,
        scheduler_checkpoint_step=100000,
        save_and_sample_every=1000,
        results_folder="./results",
        tensorboard_dir="./tensorboard_logs/diffusion-video/",
        model_name="model",
        val_num_of_batch=1,
        optimizer="adam",
        ema_decay=0.999,
        ema_update_interval=10,
        ema_step_start=100,
        use_mixed_precision=False
    ):
        super().__init__()
        self.model = diffusion_model
        self.val_num_of_batch = val_num_of_batch
        self.sample_steps = sample_steps
        self.save_and_sample_every = save_and_sample_every

        self.train_num_steps = train_num_steps

        # self.train_dl_class = train_dl
        # self.val_dl_class = val_dl
        self.train_dl = train_dl
        self.val_dl = val_dl
        if optimizer == "adam":
            self.opt = Adam(self.model.parameters(), lr=train_lr)
        elif optimizer == "adamw":
            self.opt = AdamW(self.model.parameters(), lr=train_lr)
        self.scheduler = LambdaLR(self.opt, lr_lambda=scheduler_function)
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_interval, power=0.75, update_after_step=ema_step_start)
        if use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.step = 0
        self.device = rank
        self.scheduler_checkpoint_step = scheduler_checkpoint_step

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        self.writer = SummaryWriter(tensorboard_dir)
        train_mean, train_std = cal_mean_std(train_dl, train_dl.__len__())
        self.train_mean, self.train_std = train_mean.unsqueeze(0).to(rank), train_std.unsqueeze(0).to(rank)
        val_mean, val_std = cal_mean_std(val_dl, val_dl.__len__())
        self.val_mean, self.val_std = val_mean.unsqueeze(0).to(rank), val_std.unsqueeze(0).to(rank)


    def save(self):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
        }
        idx = (self.step // self.save_and_sample_every) % 3
        torch.save(data, str(self.results_folder / f"{self.model_name}_{idx}.pt"))

    def load(self, idx=0, load_step=True):
        data = torch.load(
            str(self.results_folder / f"{self.model_name}_{idx}.pt"),
            map_location=lambda storage, loc: storage,
        )
        print(str(self.results_folder / f"{self.model_name}_{idx}.pt"))
        all_params = data["model"].keys()
        poped_params = []
        for key in all_params:
            if "train_" in key:
                poped_params.append(key)
        for key in poped_params:
            data["model"].pop(key)
        if "ema" in data.keys():
            all_params = data["ema"].keys()
            poped_params = []
            for key in all_params:
                if "train_" in key:
                    poped_params.append(key)
            for key in poped_params:
                data["ema"].pop(key)
        if load_step:
            self.step = data["step"]

        self.model.load_state_dict(data["model"], strict=False)
        if "ema" not in data.keys():
            self.ema.ema_model.load_state_dict(data["model"], strict=False)
        else:
            self.ema.load_state_dict(data["ema"], strict=False)
    
    def normalize(self, data, mean, std):
        normalized_data = (data - mean) / std
        return normalized_data

    def train(self):
        pbar = tqdm(total=self.train_num_steps, initial=self.step)
        while self.step < self.train_num_steps:
            if (self.step >= self.scheduler_checkpoint_step) and (self.step != 0):
                self.scheduler.step()
            train_loss = []
            for data in self.train_dl:
                self.opt.zero_grad()
                data = data.to(self.device)
                data_norm = self.normalize(data, self.train_mean, self.train_std)
                self.model.train()
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss, aloss = self.model(data_norm)
                    self.scaler.scale(loss).backward()
                    self.scaler.scale(aloss).backward()
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss, aloss = self.model(data_norm)
                    loss.backward()
                    aloss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                train_loss.append(loss.item())
                
            t_loss = sum(train_loss) / len(train_loss)
            self.writer.add_scalar("loss", t_loss, self.step)
            self.ema.update()

            if (self.step % self.save_and_sample_every == 0):
                bpps = []
                mses = []
                for i, batch in enumerate(self.val_dl):
                    batch = batch.to(self.device)
                    batch_norm = self.normalize(batch, self.val_mean, self.val_std)
                    batch_norm = batch_norm.to(self.device)
                    if i >= self.val_num_of_batch:
                        break
                    self.ema.ema_model.eval()
                    compressed, bpp = self.ema.ema_model.compress(
                        batch_norm, self.sample_steps
                    )
                    bpps.append(bpp)
                    mse = torch.nn.functional.mse_loss(batch, compressed * self.val_std + self.val_mean)
                    mses.append(mse)
                self.writer.add_scalar(
                    f"mse/num{i}",
                    sum(mses) / len(mses),
                    self.step // self.save_and_sample_every,
                )
                self.writer.add_scalar(
                    f"bpp/num{i}",
                    sum(bpps) / len(bpps),
                    self.step // self.save_and_sample_every,
                )
                self.save()

            self.step += 1
            pbar.update(1)
        self.save()
        pbar.n = self.train_num_steps
        pbar.close()
        print("training completed")
